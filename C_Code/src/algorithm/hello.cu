#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel function that will run on the GPU
__global__ void hello_kernel() {
    // Get the unique thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU thread %d!\n", tid);
}

// Function to check CUDA errors
void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", message, cudaGetErrorString(error));
        
    }
}

int main() {
    // Print from CPU (host)
    printf("Hello from CPU!\n");

    // Get GPU device properties
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, 0);  // Get properties of device 0
    checkCudaError(error, "Failed to get device properties");

    // Print some basic GPU information
    printf("\nGPU Device Properties:\n");
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    // Launch configuration
    int numThreads = 4;    // Number of threads per block
    int numBlocks = 2;     // Number of blocks

    // Launch CUDA kernel
    printf("\nLaunching GPU kernel...\n");
    hello_kernel<<<numBlocks, numThreads>>>();

    // Wait for GPU to finish and check for errors
    error = cudaDeviceSynchronize();
    checkCudaError(error, "Kernel launch failed");

    // Reset device before exit
    cudaDeviceReset();

    return 0;
}