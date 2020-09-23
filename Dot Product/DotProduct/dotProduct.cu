#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef __CUDACC__  
#define __CUDACC__
#endif
#define N 1//1024

__global__ void dotProduct(float *out, float *a, float *b)
{
	//Shared memory for multiplication
	__shared__ float cache[N];
	cache[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

	//synchronise threads
	__syncthreads();

	//Use thread 0 to sum the products
	if (threadIdx.x == 0)
	{		
		float sum = 0.0;
		for (int i = 0; i < N; i++)
		{
			sum += cache[i];
		}	
		//printf("%f\n", sum);
		*out = sum;
	}
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
	out = (float*)malloc(sizeof(float));
	out[0] = 1.0f;
	for (int i = 0; i < N; i++)
	{
		a[i] = 3.0f;
		b[i] = 5.0f;
	}
	//Allocate device memeory for a
	cudaMalloc((void**)&d_a, sizeof(float) * N);
	//Transfer from host memory to device memory
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	//Allocate device memeory for b
	cudaMalloc((void**)&d_b, sizeof(float) * N);
	//Transfer from host memory to device memory
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
	//Allocate device memeory for out
	cudaMalloc((void**)&d_out, sizeof(float));
	//Transfer from host memory to device memory
	cudaMemcpy(d_out, out, sizeof(float), cudaMemcpyHostToDevice);
	//Run function in parallel in 1024 threads
	//printf("%f\n", a[0]);
	//printf("%f\n", b[0]);
	//printf("%f\n", out[0]);
	dotProduct<<<1, N>>>(d_out, d_a, d_b);
	cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
	//printf("Device %f\n", d_out[0]);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
	free(a);
	free(b);
	free(out);
	return 0;
}