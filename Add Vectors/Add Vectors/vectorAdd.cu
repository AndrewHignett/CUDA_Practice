
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define N 1000000000

__global__ void vectorAdd(float *out, float *a, float *b, int n)
{
	//Only calculate if the threadID refers to an existing element
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < n)
	{
		out[threadID] = a[threadID] + b[threadID];
	}
	//no need to return anything since the contents of the memory location have been modified
}

int main()
{
	float *a, *b, *out;
	float *d_a, *d_b, *d_out;
	//Allocate host memory for a
	a = (float*)malloc(sizeof(float) * N);
	//Allocate host memory for b
	b = (float*)malloc(sizeof(float) * N);
	//Allocate host memory for out
	out = (float*)malloc(sizeof(float) * N);
	for (int i = 0; i < N; i++)
	{
		a[i] = 1.0f;
		b[i] = 2.0f;
	}
	//Allocate device memory for a
	cudaMalloc((void**)&d_a, sizeof(float) * N);
	//Transfer from host to device memory
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	//Allocate device memory for b
	cudaMalloc((void**)&d_b, sizeof(float) * N);
	//Transfer from host to device memory
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
	//Allocate device memory for out
	cudaMalloc((void**)&d_out, sizeof(float) * N);
	//Run function in parallel in 1024 threads
	vectorAdd<<<56,1024>>>(d_out, d_a, d_b, N);
	cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
	free(a);
	free(b);
	free(out);
    return 0;
}

/*
1. Allocate host memory and initialized host data
2. Allocate device memory
3. Transfer input data from host to device memory
4. Execute kernels
5. Transfer output from device memory to host
*/