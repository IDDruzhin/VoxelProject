#pragma once

struct RenderingCB
{
	Matrix worldViewProj;
	Matrix worldView;
	float stepSize;
	float stepRatio;
	float randomX;
	float randomY;
};
const int RenderingCBAlignedSize = (sizeof(RenderingCB) + 255) & ~255;

struct ComputeBlocksCB
{
	int4 min;
	int4 max;
	int4 dim;
	int4 dimBlocks;
	int voxelsCount;
	int blockSize;
	int computeBlocksCount;
	int overlap;
	Matrix bones[256];
};
const int ComputeBlocksCBAlignedSize = (sizeof(ComputeBlocksCB) + 255) & ~255;

struct RenderBonesCB
{
	Matrix bones[256];
};
const int RenderBonesCBAlignedSize = (sizeof(RenderBonesCB) + 255) & ~255;

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
	BYTE bone00;
	BYTE bone01;
	Voxel(){};
	Voxel(UINT _index, BYTE _color, BYTE _segment) : index(_index), color(_color), segment(_segment) {};
};

struct Block
{
	Vertex v[8];
	Block() {};
	Block(int3 min, int3 max, int overlap)
	{
		Vector3 tex_min(min.x - overlap, min.y - overlap, min.z - overlap);
		Vector3 tex_max(max.x + 1 + overlap, max.y + 1 + overlap, max.z + 1 + overlap);
		Vector3 p = Vector3(min.x, min.y, min.z);
		Vector3 t = (p - tex_min);
		v[0] = { p, t };
		p = Vector3(min.x, min.y, max.z + 1);
		t = (p - tex_min);
		v[1] = { p, t };
		p = Vector3(min.x, max.y + 1, min.z);
		t = (p - tex_min);
		v[2] = { p, t };
		p = Vector3(min.x, max.y + 1, max.z + 1);
		t = (p - tex_min);
		v[3] = { p, t };
		p = Vector3(max.x + 1, min.y, min.z);
		t = (p - tex_min);
		v[4] = { p, t };
		p = Vector3(max.x + 1, min.y, max.z + 1);
		t = (p - tex_min);
		v[5] = { p, t };
		p = Vector3(max.x + 1, max.y + 1, min.z);
		t = (p - tex_min);
		v[6] = { p, t };
		p = Vector3(max.x + 1, max.y + 1, max.z + 1);
		t = (p - tex_min);
		v[7] = { p, t };
	};
};

struct BlockInfo
{
	int3 min;
	int3 max;
};

struct BlockPriorityInfo
{
	int blockIndex;
	int3 block3dIndex;
	int priority;
};

struct BlockPositionInfo
{
	int3 block3dIndex;
	Vector3 position;
	float distance;
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
};

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
	};
};
