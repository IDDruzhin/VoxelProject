#pragma once

struct RenderingCB
{
	Matrix worldViewProj;
};
const int RenderingCBAlignedSize = (sizeof(RenderingCB) + 255) & ~255;

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
};

struct Block
{
	/*
	Vertex v[8];
	Block(Vector3 dim, Vector3 index, Vector3 startPos, float blockSize)
	{
		Vector3 p = Vector3(0.0f,0.0f,0.0f);
		Vector3 t = Vector3(1,1,1);
		v[0] = { p,t };
		p = Vector3(0.0f, 0.0f, 1.0f);
		v[1] = { p,t };
		p = Vector3(0.0f, 1.0f, 0.0f);
		v[2] = { p,t };
		p = Vector3(0.0f, 1.0f, 1.0f);
		v[3] = { p,t };
		p = Vector3(1.0f, 0.0f, 0.0f);
		v[4] = { p,t };
		p = Vector3(1.0f, 0.0f, 1.0f);
		v[5] = { p,t };
		p = Vector3(1.0f, 1.0f, 0.0f);
		v[6] = { p,t };
		p = Vector3(1.0f, 1.0f, 1.0f);
		v[7] = { p,t };
	}
	*/
	/*
	Vertex v[8];
	Block(Vector3 dim, Vector3 index, Vector3 startPos, int blockDim, float blockSize)
	{
		Vector3 p = startPos + Vector3(index.x, index.y, index.z) * blockSize;
		Vector3 t = Vector3(index.x, index.y, index.z)*blockDim / dim;
		v[0] = { p,t };
		p = startPos + Vector3(index.x, index.y, index.z+1) * blockSize;
		t = Vector3(index.x, index.y, index.z+1)*blockDim / dim;
		v[1] = { p,t };
		p = startPos + Vector3(index.x, index.y+1, index.z) * blockSize;
		t = Vector3(index.x, index.y+1, index.z)*blockDim / dim;
		v[2] = { p,t };
		p = startPos + Vector3(index.x, index.y+1, index.z + 1) * blockSize;
		t = Vector3(index.x, index.y+1, index.z + 1)*blockDim / dim;
		v[3] = { p,t };
		p = startPos + Vector3(index.x+1, index.y, index.z) * blockSize;
		t = Vector3(index.x+1, index.y, index.z)*blockDim / dim;
		v[4] = { p,t };
		p = startPos + Vector3(index.x+1, index.y, index.z + 1) * blockSize;
		t = Vector3(index.x+1, index.y, index.z + 1)*blockDim / dim;
		v[5] = { p,t };
		p = startPos + Vector3(index.x+1, index.y+1, index.z) * blockSize;
		t = Vector3(index.x+1, index.y+1, index.z)*blockDim / dim;
		v[6] = { p,t };
		p = startPos + Vector3(index.x+1, index.y+1, index.z + 1) * blockSize;
		t = Vector3(index.x+1, index.y+1, index.z + 1)*blockDim / dim;
		v[7] = { p,t };
	}
	*/
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

struct BlockInfo
{
	int x_min;
	int y_min;
	int z_min;

	int x_max;
	int y_max;
	int z_max;
	BlockInfo(Vector3 index, int blockDim)
	{
		x_min = index.x*blockDim;
		y_min = index.y*blockDim;
		z_min = index.z*blockDim;

		x_max = index.x*(blockDim+1);
		y_max = index.y*(blockDim + 1);
		z_max = index.z*(blockDim + 1);
	}
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
	UINT segmentIndex;
};
