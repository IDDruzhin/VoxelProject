#include "stdafx.h"
#include "Structures.h"

bool CompareVoxelsRed(const RGBVoxel &a, const RGBVoxel &b)
{
	return (a.color.x < b.color.x);
}
bool CompareVoxelsGreen(const RGBVoxel &a, const RGBVoxel &b)
{
	return (a.color.y < b.color.y);
}
bool CompareVoxelsBlue(const RGBVoxel &a, const RGBVoxel &b)
{
	return (a.color.z < b.color.z);
}

bool CompareColorsIntensity(const uchar4 &a, const uchar4 &b)
{
	return ((a.x + a.y + a.z) < (b.x + b.y + b.z));
}