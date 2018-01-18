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
	Vertex v[8];
	Block(Vector3 size, Vector3 index, Vector3 startPos, float blockSize)
	{
		Vector3 p = startPos + Vector3(index.x, index.y, index.z) * blockSize;
		Vector3 t = Vector3(index.x, index.y, index.z) / size;
		v[0] = { p,t };
		p = startPos + Vector3(index.x, index.y, index.z+1) * blockSize;
		t = Vector3(index.x, index.y, index.z+1) / size;
		v[1] = { p,t };
		p = startPos + Vector3(index.x, index.y+1, index.z) * blockSize;
		t = Vector3(index.x, index.y+1, index.z) / size;
		v[2] = { p,t };
		p = startPos + Vector3(index.x, index.y+1, index.z + 1) * blockSize;
		t = Vector3(index.x, index.y+1, index.z + 1) / size;
		v[3] = { p,t };
		p = startPos + Vector3(index.x+1, index.y, index.z) * blockSize;
		t = Vector3(index.x+1, index.y, index.z) / size;
		v[4] = { p,t };
		p = startPos + Vector3(index.x+1, index.y, index.z + 1) * blockSize;
		t = Vector3(index.x+1, index.y, index.z + 1) / size;
		v[5] = { p,t };
		p = startPos + Vector3(index.x+1, index.y+1, index.z) * blockSize;
		t = Vector3(index.x+1, index.y+1, index.z) / size;
		v[6] = { p,t };
		p = startPos + Vector3(index.x+1, index.y+1, index.z + 1) * blockSize;
		t = Vector3(index.x+1, index.y+1, index.z + 1) / size;
		v[7] = { p,t };
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