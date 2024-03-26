#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#include <device_functions.h>

#define WIDTH (1500) //CHANGE THIS!!!

#define TILE_WIDTH 2 //CHANGE THIS!!! [2,5,10,25]

//Tiled Multiplication Kernel
__global__ void matrixMulKernel(float* M, float* N, float* P, int Width) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

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

void matrixMulCPU(float* M, float* N, float* P, int Width) {
	for (int i = 0; i < Width; ++i) {
		for (int j = 0; j < Width; ++j) {
			float sum = 0;
			for (int k = 0; k < Width; ++k) {
				sum += M[i * Width + k] * N[k * Width + j];
			}
			P[i * Width + j] = sum;
		}
	}
}

int main() {
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

	int NumBlocks = WIDTH / TILE_WIDTH;
	if (WIDTH % TILE_WIDTH) NumBlocks++;

	dim3 dimGrid(NumBlocks, NumBlocks);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float elapsedTime_MatrixMulTiled;

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

	cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);

	matrixMulKernel << <dimGrid, dimBlock, 0, 0 >> > (d_M, d_N, d_P, WIDTH);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime_MatrixMulTiled, start, stop);
	printf("Device Matrix Mul Time, size[%d]: %f ms\n", WIDTH, elapsedTime_MatrixMulTiled);

	cudaMemcpy(h_M, d_M, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_N, d_N, size, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

	//Do CPU matrix multiplication to refer to
	matrixMulCPU(h_M, h_N, h_Pcheck, WIDTH);

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