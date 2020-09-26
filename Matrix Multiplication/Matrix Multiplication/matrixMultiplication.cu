
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define M 2
#define N 4
#define P 3
#define BLOCKS (M * P - 1)/1024 + 1

__global__ void matrixMultiply(int *out, int *matrixA, int *matrixB)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < N * P)
	{
		int row = threadId % M;
		int column = threadId % P;
		int sum = 0;
		for (int i = 0; i < N; i++)
		{
			//printf("ThreadId %d - %d\nRow: %d\nColumn: %d\nA: %d\nB: %d\n", threadId, i, row, column, *(matrixA + i), *(matrixB + i*P));
			sum += *(matrixA + i) * *(matrixB + i * P);
		}
		*(out + column * M + row) = sum;
	}
}

int main()
{
	int *matrixA = (int*)malloc(M * N * sizeof(int));
	int *matrixB = (int*)malloc(N * P * sizeof(int));
	int *out = (int*)malloc(M * P * sizeof(int));
	int *d_matrixA, *d_matrixB, *d_out;
	int threadCount = 1024;
	if (BLOCKS == 1)
	{
		threadCount = M * P;
	}

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			*(matrixA + i * N + j) = rand();
			printf("%d ", *(matrixA + i * N + j));
		}
		printf("\n");
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < P; j++)
		{
			*(matrixB + i * P + j) = rand();
			printf("%d ", *(matrixB + i * P + j));
		}
		printf("\n");
	}

	//allocate device memory for matrix A
	cudaMalloc((void**)&d_matrixA, M * N * sizeof(int));
	//transfer matrix A from host to device memory
	cudaMemcpy(d_matrixA, matrixA, M * N * sizeof(int), cudaMemcpyHostToDevice);
	//allocate device memory for matrix B
	cudaMalloc((void**)&d_matrixB, N * P * sizeof(int));
	//transfer matrix B from host to device memory
	cudaMemcpy(d_matrixB, matrixB, N * P * sizeof(int), cudaMemcpyHostToDevice);
	//allocate device memory for output
	cudaMalloc((void**)&d_out, M * P * sizeof(int));
	//Threads are N * P, but split into blocks, where appropriate
	matrixMultiply<<<BLOCKS, threadCount>>>(d_out, d_matrixA, d_matrixB);
	//transfer output from device memory to host memory
	cudaMemcpy(out, d_out, M * P * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < P; j++)
		{
			printf("%d ", *(out + i * P + j));
		}
		printf("\n");
	}

	//free device memory
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_out);
	//free host memory
	free(matrixA);
	free(matrixB);
	free(out);
    return 0;
}