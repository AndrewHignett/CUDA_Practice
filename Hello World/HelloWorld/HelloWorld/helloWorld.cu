#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void helloWorld() {
	printf("Hello World\n");
}

int main() {
	helloWorld <<<1, 1>>> ();
	return 0;
}