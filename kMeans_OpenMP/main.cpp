#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <cstdlib>

//Definizione delle variabili iniziali:

#define NUM_THREADS 1   // <- Number of threads to use
#define NUM_CLUSTERS 100   // <- Number of clusters
#define NUM_POINTS 1000   // <- Number of points
#define ITER 20   // <- Number of iterations

using namespace std;

struct Point {
    double x, y;
    int cluster;
};

double distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

int main() {
    printf("****************************************************************************************************************");
    printf("\n");
    printf("************************************************ KMEANS WITH OPENMP ********************************************");
    printf("\n");
    printf("****************************************************************************************************************");
    printf("\n");
    printf("\n");
    printf("\n");
    printf("press any button to begin execution.");
    printf("\n");
    getchar();
    vector<Point> data(NUM_POINTS);
    vector<Point> centroids(NUM_CLUSTERS);

    // Initialize data points
    for (int i = 0; i < NUM_POINTS; i++) {
        data[i].x = (double)rand() / 5;
        data[i].y = (double)rand() / 5;
    }

    // Initialize centroids
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        centroids[i].x = (double)rand() / 5;
        centroids[i].y = (double)rand() / 5;
    }

    //auto start = std::chrono::high_resolution_clock::now();
    clock_t begin = clock();
    int iterations = ITER;
    while (iterations--) {
#pragma omp parallel num_threads(NUM_THREADS)
        {
    //assign points to clusters
#pragma omp for
        for (int i = 0; i < NUM_POINTS; i++) {
            double cluster = 0;
            double min_distance = distance(data[i], centroids[0]);
            for (int j = 1; j < NUM_CLUSTERS; j++) {
                double dist = distance(data[i], centroids[j]);
                if (dist < min_distance) {
                     min_distance = dist;
                     cluster = j;
                 }
             }
             data[i].cluster = cluster;
        }
    //recalculate centroids
#pragma omp for
            for (int i = 0; i < NUM_CLUSTERS; i++) {
                int count = 0;
                Point sum = { 0, 0 };
                for (int j = 0; j < NUM_POINTS; j++) {
                    if (data[j].cluster == i) {
                        sum.x += data[j].x;
                        sum.y += data[j].y;
                        count++;
                    }
                }
                if (count > 0) {
                    centroids[i].x = sum.x / count;
                    centroids[i].y = sum.y / count;
                }
            }
        }
        printf("|");
    }
    
    printf("\n");
    clock_t end = clock();
    float time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed time : %f ms", time_spent);
    printf("\n");
    // Write the data points to a file
    printf("Writing data to file, please wait...");
    printf("\n");
    ofstream file("data.dat");
    for (int i = 0; i < NUM_POINTS; i++) {
        file << data[i].x << " " << data[i].y << " " << data[i].cluster << endl;
    }
    file.close();
    printf("File is ready, press any key to plot data.");
    getchar();
    // Call gnuplot to generate a scatter plot of the data points
    std::system(R"(gnuplot -p -e "plot 'data.dat' using 1:2:3 with points palette notitle")");

    return 0;
}