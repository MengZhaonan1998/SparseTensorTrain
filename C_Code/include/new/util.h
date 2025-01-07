// util.h - Some general-purpose tools
#ifndef UTIL_H
#define UTIL_H

#include "core.h"
#include "external.h"

namespace util{

// Print matrix [rmask, cmask]
template<class T>
void PrintMatWindow(T* matrix, size_t row, size_t col,  
                    std::tuple<int,int> rmask, std::tuple<int,int> cmask)
{
    if (std::get<0>(rmask) < 0 || std::get<1>(rmask) >= row ||
        std::get<0>(cmask) < 0 || std::get<1>(cmask) >= col) {
            throw std::invalid_argument("Invalid input row or column mask.");
    }
    for (int i = std::get<0>(rmask); i <= std::get<1>(rmask); ++i) {
        for (int j = std::get<0>(cmask); j <= std::get<1>(cmask); ++j) {
            std::cout << matrix[i * col + j] << " ";        
        }
        std::cout << "\n";
    }
    return;
}

// Print a 1-dimensional array
template<class T>
void Print1DArray(T* array, size_t N) 
{
    std::cout << "[" << array[0];
    for (size_t i = 1; i < N; ++i) {
        std::cout << ", " << array[i];
    }
    std::cout << "]" << std::endl;
    return;
}

// Random array generator
template<class T>
void generateRandomArray(T* array, int size, T minValue, T maxValue) {
    // Create a random number generator and distribution
    std::random_device rd;  // Seed source
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<> dis(minValue, maxValue);
    // Fill the vector with random numbers
    for (int i = 0; i < size; ++i) {
        array[i] = dis(gen);
    }
    return;
}

/*
// CUDA-error detection function
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
void checkCuda(cudaError_t err, const char* const func, const char* const file, int const line)
{
	if (err != cudaSuccess) 
		fprintf(stderr, "CUDA Runtime Error at: %s: %s\n %s %s\n", file, line, cudaGetErrorString(err), func);
}

// CUDA-last-error detection function
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
	cudaError_t const err{ cudaGetLastError() };
	if (err != cudaSuccess)
		fprintf(stderr, "CUDA Runtime Error at: %s: %s\n %s \n", file, line, cudaGetErrorString(err));
}
*/
}

#endif

