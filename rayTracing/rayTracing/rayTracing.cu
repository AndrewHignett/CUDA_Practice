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

__device__ float dot(float matrix1[3], float matrix2[3])
{
	return matrix1[0] * matrix2[0] + matrix1[1] * matrix2[1] + matrix1[2] * matrix2[2];
}

class camera {
public:
	//Focal point
	float focalP[3];
	//Focal length
	float focalL;
	float window[2] = { 2 , 0 };
	float up[3];
	float right[3];
	float forwards[3];
	float topLeft[3];
	float pixelSize[2];

	camera(float c_focalP[], float c_focalL, int x, int y, float c_up[], float c_right[], float c_forwards[]) {
		for (int i = 0; i < 3; i++) {
			focalP[i] = c_focalP[i];
		}
		focalL = c_focalL;
		window[1] = 2*((float)x / (float)y);
		for (int i = 0; i < 3; i++) {
			up[i] = c_up[i];
		}
		for (int i = 0; i < 3; i++) {
			right[i] = c_right[i];
		}
		for (int i = 0; i < 3; i++) {
			forwards[i] = c_forwards[i];
		}
		pixelSize[0] = window[0] / (float)y;
		pixelSize[1] = window[1] / (float)x;
		for (int i = 0; i < 3; i++) {
			topLeft[i] = focalP[i] + focalL*forwards[i] + window[0]*up[i]/2 - window[1]*right[i]/2 - pixelSize[0]*up[i]/2 + pixelSize[1]*right[i]/2;	
		}
		printf("%d %d\n", x, y);
		printf("%f %f\n", window[0], window[1]);
		printf("%f %f\n", pixelSize[0], pixelSize[1]);
	}
};

class light {
public:
	float direction[3];
	float ambientIntensity;
	float localIntensity;

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
public:
	float radius;
	float centre[3];
	float colour[3];
	float diffuseInt;
	float diffuse[3];
	float specularInt;
	float specular[3];
	float ambientInt;

	sphere(float s_radius, float s_centre[3], float s_colour[3], float s_diffuseInt, float s_specularInt, float s_specular[3]) {
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
		for (int i = 0; i < 3; i++) {
			specular[i] = s_specular[i];
		}
		ambientInt = 1 - specularInt - diffuseInt;
	}

	__device__ float sphereIntersect(float *rayPoint, float *rayDirection) {
		//subtract centre from the ray's start point
		float newRayPoint[3];
		newRayPoint[0] = rayPoint[0] - centre[0];
		newRayPoint[1] = rayPoint[1] - centre[1];
		newRayPoint[2] = rayPoint[2] - centre[2];
		float b = 2 * dot(rayDirection, newRayPoint);
		float c = dot(newRayPoint, newRayPoint) - radius * radius;
		float t = b * b - 4 * c;
		if (t > 0)
		{
			float t1 = (-b - sqrt(t))/2;
			float t2 = (-b + sqrt(t))/2;
			if (t1 > t2)
			{
				return t2;
			}
			return t1;
		}
		return INFINITY;
	}

	//can't be inside the sphere class, since it contains an array of spheres
	__device__ static bool notShadowed(int *sphereCount, sphere *spheres, int closestIndex, float *location, float *lightDirection, float *rayPoint, float *rayDirection) {
		//for (int i = 0; i < *sphereCount; i++)
		//{
			//check for intersection of light bounce with spheres, for shadowing, including self-shadowing
			//float t = 0;
			//float newLocation[3];
			//if (i != closestIndex)
			//{
			//	printf("%d\n", closestIndex);
				//float tTest = spheres[i].sphereIntersect(rayPoint, rayDirection);
				//float tTest = 0;
				//if ((tTest > 0) && (tTest < INFINITY))
				//{

				//	newLocation[0] = location[0] - lightDirection[0] / 1000;
				//	newLocation[1] = location[1] - lightDirection[1] / 1000;
				//	newLocation[2] = location[2] - lightDirection[2] / 1000;
					//t = spheres[i].sphereIntersect(location, lightDirection);
				//}
			//}
			//if ((t > 0) && (t < INFINITY))
			//{
				//If a t exists within this range for any mesh, then must be diffuse
				//if diffuse for this light, leave pixelColour alone.
				//Add on if not diffuse for any given light

				//Intersection
			//	return true;
			//}
		//}
		return false;
	}

	__device__ float* getPixelColour(float *rayPoint, float closestDistance, float *rayDirection, light *lights, int *lightCount, sphere *spheres, int *sphereCount, float *location, int closestIndex) {
		float newRayPoint[3];
		for (int i = 0; i < 3; i++)
		{
			newRayPoint[i] = rayPoint[i] - centre[i];
		}
		float x = newRayPoint[0] + closestDistance * rayDirection[0];
		float y = newRayPoint[1] + closestDistance * rayDirection[1];
		float z = newRayPoint[2] + closestDistance * rayDirection[2];
		float normal[3] = { x, y, z };
		float norm = sqrtf(x * x + y * y + z * z);
		for (int i = 0; i < 3; i++)
		{
			normal[i] = normal[i] / norm;
		}
		float pixelColour[3];
		for (int i = 0; i < 3; i++)
		{
			pixelColour[i] = colour[i] * ambientInt;
		}
		for (int lightNumber = 0; lightNumber < *lightCount; lightNumber++)
		{
			
			light *thisLight = &lights[lightNumber];
			float *lightDirection = thisLight->direction;
			
			/*
			for (int i = 0; i < *sphereCount; i++)
			{
				//check for intersection of light bounce with spheres, for shadowing, including self-shadowing
				float t = 0;
				float newLocation[3];
				if (i != closestIndex)
				{
					printf("%d\n", closestIndex);
					//float tTest = spheres[i].sphereIntersect(rayPoint, rayDirection);
					float tTest = 0;
					if ((tTest > 0) && (tTest < INFINITY))
					{

						newLocation[0] = location[0] - lightDirection[0] / 1000;
						newLocation[1] = location[1] - lightDirection[1] / 1000;
						newLocation[2] = location[2] - lightDirection[2] / 1000;
						//t = spheres[i].sphereIntersect(location, lightDirection);
					}
				}
				if ((t > 0) && (t < INFINITY))
				{
					//If a t exists within this range for any mesh, then must be diffuse
					//if diffuse for this light, leave pixelColour alone.
					//Add on if not diffuse for any given light
				}
			}
			*/
			//notShadowed(sphereCount, spheres, closestIndex, location, lightDirection, rayPoint, rayDirection);
			
			
			float normLight[3];
			float lightNormal = sqrtf(lightDirection[0]*lightDirection[0] + lightDirection[1] * lightDirection[1] + lightDirection[2] * lightDirection[2]);
			float cos = dot(thisLight->direction, normal)/(lightNormal*norm);
			float reflectionVector[3];
			for (int i = 0; i < 3; i++)
			{
				pixelColour[i] += cos*diffuseInt*diffuse[i];
				reflectionVector[i] = lightDirection[i] - 2 * normal[i] * dot(normal, lightDirection);
			}
			//Add specular
			int n = 40;
			float thisSpecular[3];
			for (int i = 0; i < 3; i++)
			{
				thisSpecular[i] = specular[i] * specularInt*pow(dot(reflectionVector, rayDirection), n);
				pixelColour[i] += thisSpecular[i];
			}
		}
		return pixelColour;
	}

	__device__ float *getGlobalLocation(float *rayPoint, float closestDistance, float *rayDirection) {
		float newRayPoint[3];
		float position[3];
		for (int i = 0; i < 3; i++)
		{
			newRayPoint[i] = rayPoint[i] - centre[i];
			position[i] = newRayPoint[i] + closestDistance * rayDirection[i];
		}
		return position;
	}
};

__device__ float* rayTrace(sphere *spheres, int *sphereCount, float *rayPoint, float *rayDirection, light *lights, int *lightCount)
{
	float backgroundColour[3] = { 51, 51, 51 };
	float closestDistance = INFINITY;
	int closestIndex = -1;
	for (int i = 0; i < *sphereCount; i++)
	{
		float t = spheres[i].sphereIntersect(rayPoint, rayDirection);
		if ((0 < t) && (t < closestDistance))
		{
			closestDistance = t;
			closestIndex = i;
		}
	}
	if (closestIndex == -1)
	{
		return backgroundColour;
	}
	float *location = spheres[closestIndex].getGlobalLocation(rayPoint, closestDistance, rayDirection);
	float *tempColour = spheres[closestIndex].getPixelColour(rayPoint, closestDistance, rayDirection, lights, lightCount, spheres, sphereCount, location, closestIndex);
	/*
	for (int i = 0; i < *sphereCount; i++)
	{
		//check for intersection of light bounce with spheres, for shadowing, including self-shadowing
		for (int j = 0; j < 3; j++){
			location[j] += spheres[i].centre[j];
		}
		t = spheres[i].sphereIntersect(location - lightDirection, lightDirection);

	}
	*/
	return tempColour;
}

//lightCount and sphereCount could maybe be replaced with sizeof(spheres)/sizeof(sphere)
__global__ void getPixel(float *out, camera *cam, light *lights, int *x, int *y, sphere *spheres, int *lightCount, int *sphereCount)
{
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	if ((c < *x) && (r < *y))
	{
		float givenPixel[3];
		float rayPoint[3];
		float rayDirection[3];
		for (int i = 0; i < 3; i++)
		{
			givenPixel[i] = cam->topLeft[i] - r * cam->pixelSize[0] * cam->up[i] + c * cam->pixelSize[1] * cam->right[i];
			rayPoint[i] = cam->focalP[i];
			rayDirection[i] = givenPixel[i] - rayPoint[i];
		}
		//Normalise the ray direction
		float directionNorm = sqrtf(rayDirection[0] * rayDirection[0] + rayDirection[1] * rayDirection[1] + rayDirection[2] * rayDirection[2]);
		rayDirection[0] = rayDirection[0] / directionNorm;
		rayDirection[1] = rayDirection[1] / directionNorm;
		rayDirection[2] = rayDirection[2] / directionNorm;

		if ((c == 1081) && (r == 720))
		{
			printf("%d %d %f %f %f\n", c, r, rayDirection[0], rayDirection[1], rayDirection[2]);
		}
		if ((c == 0) && (r == 720))
		{
			printf("%d %d %f %f %f\n", c, r, rayDirection[0], rayDirection[1], rayDirection[2]);
		}
		float *colour = rayTrace(spheres, sphereCount, rayPoint, rayDirection, lights, lightCount);
		if ((c == 1090) && (r == 720))
		{
			printf("%f %f %f\n", colour[0], colour[1], colour[2]);
		}
		for (int i = 0; i < 3; i++)
		{
			*(out + c * *y * 3 + r * 3 + i) = *(colour + i);
			if ((r == 720)&&(c == 1090))
			{
				printf("%d %d %d %d\n", r, c, i, c * *y * 3 + r * 3 + i);
				printf("%f\n", *(colour + i));
			}	
		}
	}
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
	//x and y swapped for priting mat
	int *x = (int*)malloc(sizeof(int));
	*x = 2160;
	int *y = (int*)malloc(sizeof(int));
	*y = 3840;
	float focalP[3] = {0, 0, 0};
	float focalL = 1;
	float up[3] = { 0, 1, 0 };
	float right[3] = { 1, 0, 0 };
	float forwards[3] = { 0, 0, 1 };
	camera *camera1 = (camera*)malloc(sizeof(camera));
	//camera
	*camera1 = camera(focalP, focalL, *x, *y, up, right, forwards);
	float lightDirection1[] = { -2, 1, -3 };
	float ambientInt1 = 0.2;
	float localInt1 = 0.8;
	int *lightCount = (int*)malloc(sizeof(int));
	*lightCount = 1;
	light *lights = (light*)malloc(*lightCount * sizeof(light));
	//light
	light light1 = light(lightDirection1, ambientInt1, localInt1);
	*lights = { light1 };
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
				*(out + i * *y * 3 + j * 3 + k) = (float)0;
				//printf("%d\n", i * *y * 3 + j * 3 + k);
			}
		}
	}

	float radius1 = 0.5;
	float centre1[3] = {0, 0, 1.5};
	float colour1[3] = {0, 0, 255};
	float diffuseInt1 = 0.4;
	float specularInt1 = 1;
	float specular1[3] = {255, 255, 255};
	int *sphereCount = (int*)malloc(sizeof(int));
	*sphereCount = 1;
	sphere *spheres = (sphere*)malloc(*sphereCount * sizeof(sphere));
	sphere sphere1 = sphere(radius1, centre1, colour1, diffuseInt1, specularInt1, specular1);
	*spheres = { sphere1 };

	int *d_x, *d_y, *d_lightCount, *d_sphereCount;
	float *d_out;
	camera *d_camera1;
	light *d_lights;
	sphere *d_spheres;
	//allocate device memory for variables
	cudaMalloc((void**)&d_x, sizeof(int));
	cudaMalloc((void**)&d_y, sizeof(int));
	cudaMalloc((void**)&d_out, *x * *y * 3 * sizeof(float));
	cudaMalloc((void**)&d_camera1, sizeof(camera));
	cudaMalloc((void**)&d_lights, *lightCount * sizeof(light));
	cudaMalloc((void**)&d_spheres, *sphereCount * sizeof(sphere));
	cudaMalloc((void**)&d_lightCount, sizeof(int));
	cudaMalloc((void**)&d_sphereCount, sizeof(int));
	//transfer from host to device memory
	cudaMemcpy(d_x, x, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_camera1, camera1, sizeof(camera), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lights, lights, *lightCount * sizeof(light), cudaMemcpyHostToDevice);
	cudaMemcpy(d_spheres, spheres, *sphereCount * sizeof(sphere), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lightCount, lightCount, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sphereCount, sphereCount, sizeof(int), cudaMemcpyHostToDevice);
	getPixel<<<grid, block>>>(d_out, d_camera1, d_lights, d_x, d_y, d_spheres, d_lightCount, d_sphereCount);
	//transfer output from device memory to host memory
	cudaMemcpy(out, d_out, *x * *y * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//free device memory
	cudaFree(d_camera1);
	cudaFree(d_lights);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_out);
	cudaFree(d_spheres);
	cudaFree(d_lightCount);
	cudaFree(d_sphereCount);
	//Opencv documentation approach to saving an image
	Mat mat(*x, *y, CV_8UC4);
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
	free(lights);
	free(out);
	free(spheres);
	free(lightCount);
	free(sphereCount);
	return 0;
}