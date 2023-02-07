cmake_minimum_required(VERSION 3.17)
project(KMeans)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -openmp")
add_executable(KMeans main.cpp)