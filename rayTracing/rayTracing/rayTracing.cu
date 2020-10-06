#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#include <opencv2\opencv.hpp>
#include <opencv2\core\core_c.h>
#include <opencv2\highgui\highgui_c.h>

using namespace cv;
using namespace std;

class camera {
	//Focal point
	float focalP[3];
	//Focal length
	float focalL;
	int window[2] = { 1 , 0 };
	float up[3];
	float right[3];
	float forwards[3];
	float topLeft[3];
	float pixelSize[2];
public:
	camera(float c_focalP[], float c_focalL, int x, int y, float c_up[], float c_right[], float c_forwards[]) {
		for (int i = 0; i < 3; i++) {
			focalP[i] = c_focalP[i];
		}
		focalL = c_focalL;
		window[1] = x / y;
		for (int i = 0; i < 3; i++) {
			up[i] = c_up[i];
		}
		for (int i = 0; i < 3; i++) {
			right[i] = c_right[i];
		}
		for (int i = 0; i < 3; i++) {
			forwards[i] = c_forwards[i];
		}
		pixelSize[0] = window[1] / y;
		pixelSize[1] = window[2] / x;
		for (int i = 0; i < 3; i++) {
			topLeft[i] = focalP[i] + focalL*forwards[i] + window[1]*up[i] - window[2]*right[i] - pixelSize[1]*up[i] + pixelSize[2]*right[i];
		}
	}
};

class light {
	float direction[3];
	float ambientIntensity;
	float localIntensity;
public:
	light(float l_direction[], float l_ambient, float l_local) {
		float norm = sqrtf(l_direction[0] * l_direction[0] + l_direction[1] * l_direction[1] + l_direction[2] * l_direction[2]);
		for (int i = 0; i < 3; i++){
			direction[i] = l_direction[i] / norm;
		}
		ambientIntensity = l_ambient;
		localIntensity = l_local;
	}
};

//custom meshes
class mesh {

};

class sphere {
	float radius;
	float centre[3];
	float colour[3];
	float diffuseInt;
	float diffuse[3];
	float specularInt;
	float specular;
	float ambientInt;
public:
	sphere(float s_radius, float s_centre[3], float s_colour[3], float s_diffuseInt, float s_specularInt, float specular) {
		radius = s_radius;
		for (int i = 0; i < 3; i++) {
			centre[i] = s_centre[i];
		}
		for (int i = 0; i < 3; i++) {
			colour[i] = s_colour[i];
		}
		diffuseInt = s_diffuseInt;
		for (int i = 0; i < 3; i++) {
			diffuse[i] = s_colour[i];
		}
		specularInt = s_specularInt;
		specular = specular;
		ambientInt = 1 - specularInt - diffuseInt;
	}
};

__device__ int* getColour()
{

}

__global__ void getPixel(float *out, camera *cam, light *light, int *x, int *y)
{
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	if ((c < *x) && (r < *y))
	{
		for (int i = 0; i < 3; i++)
		{
			float PLACEHOLDER = 128;
			*(out + c * *y * 3 + r * 3 + i) = PLACEHOLDER;
		}
	}
	//__syncthreads();
}

void makeImage(Mat &mat, float *image, int *y)
{
	CV_Assert(mat.channels() == 4);
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			Vec4b& bgra = mat.at<Vec4b>(i, j);
			bgra[0] = saturate_cast<uchar>(*(image + i * *y * 3 + j * 3)); //Blue
			bgra[1] = saturate_cast<uchar>(*(image + i * *y * 3 + j * 3 + 1)); //Green
			bgra[2] = saturate_cast<uchar>(*(image + i * *y * 3 + j * 3 + 2)); //Red
			bgra[3] = saturate_cast<uchar>((float)255.0); //Alpha
		}
	}
}

int main()
{
	int *x = (int*)malloc(sizeof(int));
	*x = 1920;
	int *y = (int*)malloc(sizeof(int));
	*y = 1080;
	float focalP[] = {0, 0, 0};
	float focalL = 1;
	float up[3] = { 0, 1, 0 };
	float right[3] = { 1, 0, 0 };
	float forwards[] = { 0, 0, 1 };
	camera *camera1 = (camera*)malloc(sizeof(camera));
	//camera
	*camera1 = camera(focalP, focalL, *x, *y, up, right, forwards);
	float lightDirection[] = { -2, 1, -3 };
	float ambientInt = 0.2;
	float localInt = 0.8;
	light *light1 = (light*)malloc(sizeof(light));
	//light
	*light1 = light(lightDirection, ambientInt, localInt);
	//limited 1024
	dim3 block(32, 32, 1);
	dim3 grid;
	grid.x = (*x + block.x - 1) / block.x;
	grid.y = (*y + block.y - 1) / block.y;
	float *out = (float*)malloc(*x * *y * 3 * sizeof(float));
	for (int i = 0; i < *x; i++)
	{
		for (int j = 0; j < *y; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				*(out + i * *y * 3 + j * 3 + k) = (float)51;
			}
		}
	}
	int *d_x, *d_y;
	float *d_out;
	camera *d_camera1;
	light *d_light1;
	//allocate device memory for variables
	cudaMalloc((void**)&d_x, sizeof(int));
	cudaMalloc((void**)&d_y, sizeof(int));
	cudaMalloc((void**)&d_out, *x * *y * 3 * sizeof(float));
	cudaMalloc((void**)&d_camera1, sizeof(camera));
	cudaMalloc((void**)&d_light1, sizeof(light));
	//transfer from host to device memory
	cudaMemcpy(d_x, x, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_camera1, camera1, sizeof(camera), cudaMemcpyHostToDevice);
	cudaMemcpy(d_light1, light1, sizeof(light), cudaMemcpyHostToDevice);
	getPixel<<<grid, block>>>(d_out, d_camera1, d_light1, d_x, d_y);
	//transfer output from device memory to host memory
	cudaMemcpy(out, d_out, *x * *y * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//free device memory
	cudaFree(d_camera1);
	cudaFree(d_light1);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_out);
	//Opencv documentation approach to saving an image
	Mat mat(*y, *x, CV_8UC4);
	makeImage(mat, out, y);
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	try {
		imwrite("output.png", mat, compression_params);
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}
	fprintf(stdout, "Saved PNG file with alpha data.\n");
	//free host memory
	free(x);
	free(y);
	free(camera1);
	free(light1);
	free(out);
	return 0;
}