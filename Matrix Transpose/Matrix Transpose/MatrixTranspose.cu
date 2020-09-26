
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define ROWS 17280
#define COLS 30720
#define BLOCKS (ROWS*COLS - 1)/1024 + 1

//out is [COLS][ROWS] and matrix is [ROWS][COLS]
__global__ void matrixTranspose(int *out, int *matrix)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < ROWS * COLS)
	{
		int row = threadId / COLS;
		int column = threadId % COLS;
		//reverse row and column of each element to transpose
		*(out + column * ROWS + row) = *(matrix + threadId);
	}
}

int main()
{
	int *matrix = (int*)malloc(ROWS * COLS * sizeof(int));
	int *out = (int*)malloc(COLS * ROWS * sizeof(int));
	int *d_matrix, *d_out;
	int threadCount = 1024;
	if (BLOCKS == 1)
	{
		threadCount = ROWS * COLS;
	}

	for (int i = 0; i < ROWS; i++)
	{
		for (int j = 0; j < COLS; j++)
		{
			//populate the matrix with random integers
			*(matrix + i * COLS + j) = rand();
		}
	}

	//allocate device memory for the matrix
	cudaMalloc((void**)&d_matrix, ROWS * COLS * sizeof(int));
	//transfer from host memory to device memory
	cudaMemcpy(d_matrix, matrix, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);
	//allocate device memory for out
	cudaMalloc((void**)&d_out, COLS * ROWS * sizeof(int));
	//Transpose the matrix using an appropriate number of blocks and threads
	matrixTranspose<<<BLOCKS, threadCount>>>(d_out, d_matrix);
	//transfer from device memeory to host memory
	cudaMemcpy(out, d_out, COLS * ROWS * sizeof(int), cudaMemcpyDeviceToHost);
	
	//For printing out the matrix, if it is of a suitable size
	/*
	for (int i = 0; i < COLS; i++)
	{
		for (int j = 0; j < ROWS; j++)
		{
			printf("%d ", *(out + i * ROWS + j));
		}
		printf("\n");
	}
	*/

	//free device memory
	cudaFree(d_matrix);
	cudaFree(d_out);
	//free host memory
	free(matrix);
	free(out);
    return 0;
}