#include "stdafx.h"
#include "Structures.h"

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