#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#define N 1024
#define blockCount 1

__global__ void calculatePi(float *out)
{
	//Shared memeory for sum, only works with a blockCount of 1
	__shared__ float cache[N];
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	int stepSize = blockDim.x * gridDim.x;

	// y = root(1 - x^2)
	// calculate y from x = threadID/N (giving an over estimate)
	//sum of all y*x*4 = solution
	
	float x = float(1) /(blockCount*N);
	float thisX = float(threadID) / (blockCount*N);
	float y = sqrt(1 - thisX * thisX);
	float area = y * x * 4;
	printf("ThreadID: %d\nthisX: %f\nX: %f\nY: %f\nArea: %f\n", threadID, thisX, x, y, area);
	cache[threadID] = area;
	__syncthreads();
	
	if (threadIdx.x == 0)
	{
		float sum = 0.0;
		for (int i = 0; i < N; i++)
		{
			sum += cache[i];
		}
		*out = sum;
	}
}

int main()
{
	float out, *d_out;
	cudaMalloc((void**)&d_out, sizeof(float));
	cudaMemcpy(d_out, &out, sizeof(float), cudaMemcpyHostToDevice);
	//Parallel pi calculation, single block
	calculatePi<<<blockCount, N >>>(d_out);
	cudaMemcpy(&out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
	printf("%f\n", out);
	cudaFree(d_out);
	return 0;
}