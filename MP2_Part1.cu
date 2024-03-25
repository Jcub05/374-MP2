// Jacob Badali 20290739
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#define WIDTH (1500) //CHANGE THIS!!!

#define BLOCK_WIDTH 2 //CHANGE THIS!!!



//Multiplication kernel function
__global__ void mulKernel(float* M, float* N, float* P, int size) {
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int cols = blockIdx.x * blockDim.x + threadIdx.x;


    if (rows < size && cols < size) {
        float temp_sum = 0.0;
        for (int i = 0; i < size; i++) {
            temp_sum += M[rows * size + i] * N[i * size + cols];
        }
        P[rows * size + cols] = temp_sum;
    }
}


int main()
{

    float* d_M = 0;
    float* d_N = 0;
    float* d_P = 0;

    float* h_M;
    float* h_N;
    float* h_P;
    float* h_Pcheck;


    int size = WIDTH * WIDTH * sizeof(float);

    cudaMallocHost((void**)&h_M, size);
    cudaMallocHost((void**)&h_N, size);
    cudaMallocHost((void**)&h_P, size);
    cudaMallocHost((void**)&h_Pcheck, size);

    int NumBlocks = WIDTH / BLOCK_WIDTH;
    if (WIDTH % BLOCK_WIDTH) NumBlocks++;

    dim3 dimGrid(NumBlocks, NumBlocks);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsedTime_DevToHost;
    float elapsedTime_HostToDev;
    float elapsedTime_MatrixMulHost;
    float elapsedTime_MatrixMulDev;

    //Allocate appropriate memory size for each array
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    //fill host matrices
    for (int k = 0; k < WIDTH; k++) {
        for (int j = 0; j < WIDTH; j++) {
            h_M[k * WIDTH + j] = ((float)rand() / RAND_MAX) * 100.0f; // fill with rand values from 0-100
            h_N[k * WIDTH + j] = ((float)rand() / RAND_MAX) * 100.0f;
            h_P[k * WIDTH + j] = 0.0;
            h_Pcheck[k * WIDTH + j] = 0.0;
        }
    }

    

    //Cpy to dev, timer
    cudaEventRecord(start, 0);
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime_HostToDev, start, stop);
    printf("Transfer Host to Device, size[%d]: %f ms |", WIDTH, elapsedTime_HostToDev);

    cudaEventRecord(start, 0);
    cudaMemcpy(h_M, d_M, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_N, d_N, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTidme_DevToHost, start, stop);
    printf("| Transfer Device to Host, size[%d]: %f ms\n", WIDTH, elapsedTime_DevToHost);

    printf("\n");

    //Host Matrix Multiplication
    cudaEventRecord(start, 0);


    //for (int i = 0; i < WIDTH; i++) {
    //    for (int j = 0; j < WIDTH; j++) {
    //        for (int k = 0; k < WIDTH; k++) {
    //            h_Pcheck[i * WIDTH + j] += h_M[i * WIDTH + k] * h_N[k * WIDTH + j];
    //        }
    //    }
    //}

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime_MatrixMulHost, start, stop);
    printf("Host Matrix Mul Time, size[%d]: %f ms\n", WIDTH, elapsedTime_MatrixMulHost);

    //Device Matrix Multiplication
    cudaEventRecord(start, 0);
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventRecord(start, 0);

    mulKernel << <dimGrid, dimBlock, 0, 0>> > (d_M, d_N, d_P, WIDTH);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime_MatrixMulDev, start, stop);
    printf("Device Matrix Mul Time, size[%d]: %f ms\n", WIDTH, elapsedTime_MatrixMulDev);

    cudaMemcpy(h_M, d_M, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_N, d_N, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    int check = 0;
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (abs(h_P[i * WIDTH + j] - h_Pcheck[i * WIDTH + j]) > 1) {
                check = 1;
            }
        }
    }

    if (check == 0) {
        printf("TEST PASSED\n");
    }
    else {
        printf("TEST FAILED\n");
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(h_M);
    cudaFree(h_N);
    cudaFree(h_P);
    cudaFree(h_Pcheck);
}