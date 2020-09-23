
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void add(int *c, const int *a, const int *b)
{
	*c = *a + *b;
}

int main()
{
	int a, b, c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	//allocate space for device copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	a = 2;
	b = 7;

	//copy  inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	add<<<1, 1>>>(d_c, d_a, d_b);

	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	printf("%d\n", c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
