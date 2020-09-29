#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

//function callable from, and executable on, the device
__host__ float deviceFunction()
{

}

__global__ void fourierTransform()
{
    
}

int main()
{
	//program device-side fourier transform
	//compare to built-in function cuFFT
	return 0;
}
