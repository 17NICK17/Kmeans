
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>

//Here we have some defines:


#define CLUSTER_NUM 30			// <- numero di clusters
#define POINT_NUM 1000000        // <- numero di punti
#define THREADS_BLOCK 512	// <- Thread per block (GPU: GTX 1070).
#define IT_MAX 20               // <- iterazioni
#define EPSILON 0.001           // <- valore di tolleranza
								


#define COORD_MAX 100000		// <- coordinate
#define POINT_FEATURES 3		// <- caratteristiche di un punto (x,y,cluster)
#define CLUSTER_FEATURES 4		// <- caratteristiche di un cluster (center,sizex,sizey,npoints)



#define gpuErrchk(ans) { cudaErrCheck((ans), __FILE__, __LINE__); }


//Metodo usato per verificare eventuali errori relativi a CUDA
inline void cudaErrCheck(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUError: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


//Genera un numero casuale nel range indicato in COORD_MAX
float random() {
	float x;
	x = (float)rand() * (float)32767;
	x = fmod(x, COORD_MAX);
	return x;
}

//Inizializzo i points e i cluster
void init_all(float* points, float* clusters) {
	for (int i = 0; i < POINT_NUM; i++) {				// <- point: <x,y,cluster>
		points[i * POINT_FEATURES + 0] = random();
		points[i * POINT_FEATURES + 1] = random();
		points[i * POINT_FEATURES + 2] = 0;
	}

	for (int i = 0; i < CLUSTER_NUM; i++) {				//<- cluster: <centro,size_x,size_y,points>
		clusters[i * CLUSTER_FEATURES + 0] = rand() % POINT_NUM;
		clusters[i * CLUSTER_FEATURES + 1] = 0;
		clusters[i * CLUSTER_FEATURES + 2] = 0;
		clusters[i * CLUSTER_FEATURES + 3] = 0;
	}
}


//Effetuo il print dei punti su file
void write_to_file(float* points) {
	FILE* fPtr;
	char filePath[100] = { "file.dat" };
	char dataToAppend[1000];
	FILE* ff = fopen("file.dat", "w");
	fclose(ff);
	fPtr = fopen(filePath, "a");
	for (int i = 0; i < POINT_NUM; i++) {
		float x = points[i * POINT_FEATURES + 0];
		float y = points[i * POINT_FEATURES + 1];
		int cluster = points[i * POINT_FEATURES + 2];
		fprintf(fPtr, "%f %f %d\n", x, y, cluster);
	}
	fclose(fPtr);
}

//plotto i punti con gnuplot
void showPlot() {
	system(R"(gnuplot -p -e "plot 'file.dat' using 1:2:3 with points palette notitle")");
}

//Calcolo la distanza tra i cluster
__device__ float distance(float x1, float x2, float y1, float y2) {
	return sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)));
}

//ricalcolo i centroidi dei cluster
__global__ void centroids_update(float* points, float* clusters) {
	long id_cluster = threadIdx.x + blockIdx.x * blockDim.x;			//<- mapping del thread ID
	float sizeX = clusters[id_cluster * CLUSTER_FEATURES + 1];
	float sizeY = clusters[id_cluster * CLUSTER_FEATURES + 2];
	float nPoints = clusters[id_cluster * CLUSTER_FEATURES + 3];
	float newX = sizeX / nPoints;
	float newY = sizeY / nPoints;
	long cluster_center_index = (long)clusters[id_cluster * CLUSTER_FEATURES + 0];
	float x = points[cluster_center_index * POINT_FEATURES + 0];
	float y = points[cluster_center_index * POINT_FEATURES + 1];
	if (!(fabsf(x - newX) < EPSILON && fabsf(y - newY) < EPSILON)) {
		points[cluster_center_index * POINT_FEATURES + 0] = newX;
		points[cluster_center_index * POINT_FEATURES + 1] = newY;
	}
}

//Assegno i punti ai cluster
__global__ void assign_clusters(float* points, float* clusters) {
	long id_punto = threadIdx.x + blockIdx.x * blockDim.x;					// <- mapping del thread ID
	if (id_punto < POINT_NUM) {												// <- out of memory check
		float x_punto, x_cluster, y_punto, y_cluster = 0;
		x_punto = points[id_punto * POINT_FEATURES + 0];
		y_punto = points[id_punto * POINT_FEATURES + 1];
		long best_fit = 0;
		long distMax = LONG_MAX;
		for (int i = 0; i < CLUSTER_NUM; i++) {
			int cluster_index_point = clusters[i * CLUSTER_FEATURES + 0];
			x_cluster = points[cluster_index_point * POINT_FEATURES + 0];
			y_cluster = points[cluster_index_point * POINT_FEATURES + 1];
			if (distance(x_punto, x_cluster, y_punto, y_cluster) < distMax) {
				best_fit = i;
				distMax = distance(x_punto, x_cluster, y_punto, y_cluster);
			}
		}
		//Output, i assign the results:
		points[id_punto * POINT_FEATURES + 2] = best_fit;
		atomicAdd(&clusters[best_fit * CLUSTER_FEATURES + 1], x_punto);		// <- SEZIONE CRITICA: due punti potrebbero incrementare lo stesso cluster
		atomicAdd(&clusters[best_fit * CLUSTER_FEATURES + 2], y_punto);		//	                   in contemoranea
		atomicAdd(&clusters[best_fit * CLUSTER_FEATURES + 3], 1);			
	}
}


//Reset clusters
__global__ void cuda_remove_points_cluster(float* clusters) {
	//<centro, size_x, size_y, points>
	long id_cluster = threadIdx.x + blockIdx.x * blockDim.x;
	clusters[id_cluster * CLUSTER_FEATURES + 1] = 0;
	clusters[id_cluster * CLUSTER_FEATURES + 2] = 0;
	clusters[id_cluster * CLUSTER_FEATURES + 3] = 0;
}




int main()                           //<- program entry point.
{

	float* points = (float*)malloc(POINT_NUM * POINT_FEATURES * sizeof(float));
	float* clusters = (float*)malloc(CLUSTER_NUM * CLUSTER_FEATURES * sizeof(float));
	float* points_d = 0;
	float* cluster_d = 0;

	init_all(points, clusters);		//<- init clusters and points.
	printf("****************************************************************************************************************");
	printf("\n");
	printf("************************************************ KMEANS WITH CUDA ********************************************");
	printf("\n");
	printf("****************************************************************************************************************");
	printf("\n");
	printf("\n");
	printf("\n");
	printf("press any button to begin execution.");
	printf("\n");
	getchar();
	printf("\nAllocating data in gpu memory\n");
	printf("\n");
	// Call per allocare i dati in gpu
	cudaMalloc(&points_d, POINT_NUM * POINT_FEATURES * sizeof(float));
	cudaMalloc(&cluster_d, CLUSTER_NUM * CLUSTER_FEATURES * sizeof(float));
	cudaMemcpy(points_d, points, POINT_NUM * POINT_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cluster_d, clusters, CLUSTER_NUM * CLUSTER_FEATURES * sizeof(float), cudaMemcpyHostToDevice);

	clock_t begin = clock();	   //<- start timing
	for (int i = 0; i < IT_MAX; i++) {
		printf("|", i);
		//CUDA call to assign points:
		assign_clusters << < (POINT_NUM + THREADS_BLOCK - 1) / THREADS_BLOCK, THREADS_BLOCK >> > (points_d, cluster_d);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		//CUDA call to recompute centers:
		centroids_update << <1, CLUSTER_NUM >> > (points_d, cluster_d);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		//CUDA call to set each cluster to 0:
		cuda_remove_points_cluster << <1, CLUSTER_NUM >> > (cluster_d);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}
	printf("\n");
	cudaDeviceSynchronize();
	clock_t end = clock();		 //<- end timing
	float time_spent = (float)(end - begin) / CLOCKS_PER_SEC;

	//Here i take data back from GPU to PC
	cudaMemcpy(points, points_d, POINT_NUM * POINT_FEATURES * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(clusters, cluster_d, CLUSTER_NUM * CLUSTER_FEATURES * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(points_d);
	cudaFree(cluster_d);

	printf("\nElapsed time : %f ms", time_spent);
	printf("\n");
	//write to file.
	printf("\nWriting data to file, please wait...");
	printf("\n");
	write_to_file(points);
	printf("\nFile is ready, press any key to plot data.");
	getchar();
	showPlot();
	exit(0);
}

