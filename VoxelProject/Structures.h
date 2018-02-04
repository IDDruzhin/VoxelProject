#pragma once

struct RenderingCB
{
	Matrix worldViewProj;
};
const int RenderingCBAlignedSize = (sizeof(RenderingCB) + 255) & ~255;

struct ComputeBlocksCB
{
	int3 min;
	int3 max;
	int3 dim;
	int3 dimBlocks;
	int blockSize;
	int computeBlocksCount;
};
const int ComputeBlocksCBAlignedSize = (sizeof(ComputeBlocksCB) + 255) & ~255;

struct Vertex
{
	Vector3 pos;
	Vector3 tex;
};

struct Voxel
{
	UINT index;
	BYTE color;
	BYTE segment;
	//BYTE padding1
	//BYTE padding2
	Voxel(){};
	Voxel(UINT _index, BYTE _color, BYTE _segment) : index(_index), color(_color), segment(_segment) {};
};

/*
struct Block
{
	Vertex v[8];
	Block(Vector3 dim, Vector3 index, Vector3 startPos, int blockDim, float blockSize)
	{
		Vector3 p = Vector3(index.x, index.y, index.z) * blockDim;
		v[0] = { p, p/dim };
		p = Vector3(index.x, index.y, index.z + 1)*blockDim;
		v[1] = { p, p / dim };
		p = Vector3(index.x, index.y + 1, index.z)*blockDim;
		v[2] = { p, p / dim };
		p = Vector3(index.x, index.y + 1, index.z + 1)*blockDim;
		v[3] = { p, p / dim };
		p = Vector3(index.x + 1, index.y, index.z)*blockDim;
		v[4] = { p, p / dim };
		p = Vector3(index.x + 1, index.y, index.z + 1)*blockDim;
		v[5] = { p, p / dim };
		p = Vector3(index.x + 1, index.y + 1, index.z)*blockDim;
		v[6] = { p, p / dim };
		p = Vector3(index.x + 1, index.y + 1, index.z + 1)*blockDim;
		v[7] = { p, p / dim };
	}
};
*/

struct BlockInfo
{
	int3 min;
	int3 max;
};


struct SegmentData
{
	uchar3 color;
	UINT start;
	UINT finish;
};

struct RGBVoxel
{
	UINT index;
	uchar4 color;
	//UINT segment;
};


/*
__host__ __device__ bool CompareVoxelsRed(const RGBVoxel &a, const RGBVoxel &b);
__host__ __device__ bool CompareVoxelsGreen(const RGBVoxel &a, const RGBVoxel &b);
__host__ __device__ bool CompareVoxelsBlue(const RGBVoxel &a, const RGBVoxel &b);
*/
bool CompareColorsIntensity(const uchar4 &a, const uchar4 &b);

struct PaletteElement
{
typedef
	enum SORT_MODE
{
	SORT_MODE_RED = 0,
	SORT_MODE_GREEN = 1,
	SORT_MODE_BLUE = 2
} 	SORT_MODE;

	int start;
	int length;
	int level;
	SORT_MODE sortMode;
	PaletteElement(int _length) : start(0), length(_length), level(0), sortMode(SORT_MODE_RED) {};
	PaletteElement(const PaletteElement &parent, bool isFirst)
	{
		if (isFirst)
		{
			start = parent.start;
			length = parent.length / 2;
		}
		else
		{
			start = parent.start + parent.length / 2;
			length = parent.length - parent.length / 2;
		}
		level = parent.level + 1;
		switch (parent.sortMode)
		{
		case SORT_MODE_RED:
			sortMode = SORT_MODE_GREEN;
			break;
		case SORT_MODE_GREEN:
			sortMode = SORT_MODE_BLUE;
			break;
		case SORT_MODE_BLUE:
			sortMode = SORT_MODE_RED;
			break;
		}
	};
};
