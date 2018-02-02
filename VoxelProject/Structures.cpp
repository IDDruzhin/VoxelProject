#include "stdafx.h"
#include "Structures.h"

/*
__host__ __device__ bool CompareVoxelsRed(const RGBVoxel &a, const RGBVoxel &b)
{
	return (a.color.x < b.color.x);
}
__host__ __device__ bool CompareVoxelsGreen(const RGBVoxel &a, const RGBVoxel &b)
{
	return (a.color.y < b.color.y);
}
__host__ __device__ bool CompareVoxelsBlue(const RGBVoxel &a, const RGBVoxel &b)
{
	return (a.color.z < b.color.z);
}
*/

bool CompareColorsIntensity(const uchar4 &a, const uchar4 &b)
{
	int aIntens = 0;
	aIntens += a.x;
	aIntens += a.y;
	aIntens += a.z;
	int bIntens = 0;
	aIntens += b.x;
	aIntens += b.y;
	aIntens += b.z;
	return (aIntens < bIntens);
}