#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 2 // 

__global__ void matrixMulKernel(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row * Width + Col] = Pvalue;
}

int main() {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0); // Assuming device 0

    int max_shared_memory_per_SM = device_prop.sharedMemPerMultiprocessor;

    int max_blocks_per_SM = max_shared_memory_per_SM / (2 * TILE_WIDTH * TILE_WIDTH * sizeof(float));
    int threads_per_block = device_prop.maxThreadsPerBlock;
    int total_threads = max_blocks_per_SM * threads_per_block;

    printf("Total number of registers: %d\n", device_prop.regsPerBlock * TILE_WIDTH * TILE_WIDTH);
    printf("Shared memory size: %d bytes\n", 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float));
    printf("Number of blocks per streaming multiprocessor: %d\n", max_blocks_per_SM);
    printf("Total threads simultaneously scheduled/executing: %d\n", total_threads);

    return 0;
}
