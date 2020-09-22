
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 10000

void vectorAdd(float *out, float *a, float *b, int n)
{
	for (int i = 0; i < n; i++)
	{
		out[i] = a[i] + b[i];
	}
	//no need to return anything since the contents of the memory location have been modified
}

int main()
{
	float *a, *b, *out;
	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	out = (float*)malloc(sizeof(float) * N);
	for (int i = 0; i < N; i++)
	{
		a[i] = 1.0f;
		b[i] = 2.0f;
	}
	vectorAdd(out, a, b, N);
    return 0;
}