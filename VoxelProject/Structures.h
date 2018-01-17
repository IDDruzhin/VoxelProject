#pragma once


struct Vertex
{
	Vector3 pos;
	Vector3 tex;
};

struct Block
{
	Vertex v[8];
	Block(Vector3 size, Vector3 index, Vector3 start_pos, float block_size)
	{
		Vector3 p = start_pos + Vector3(index.x, index.y, index.z) * block_size;
		Vector3 t = Vector3(index.x, index.y, index.z) / size;
		v[0] = { p,t };
		p = start_pos + Vector3(index.x, index.y, index.z+1) * block_size;
		t = Vector3(index.x, index.y, index.z+1) / size;
		v[1] = { p,t };
		p = start_pos + Vector3(index.x, index.y+1, index.z) * block_size;
		t = Vector3(index.x, index.y+1, index.z) / size;
		v[2] = { p,t };
		p = start_pos + Vector3(index.x, index.y+1, index.z + 1) * block_size;
		t = Vector3(index.x, index.y+1, index.z + 1) / size;
		v[3] = { p,t };
		p = start_pos + Vector3(index.x+1, index.y, index.z) * block_size;
		t = Vector3(index.x+1, index.y, index.z) / size;
		v[4] = { p,t };
		p = start_pos + Vector3(index.x+1, index.y, index.z + 1) * block_size;
		t = Vector3(index.x+1, index.y, index.z + 1) / size;
		v[5] = { p,t };
		p = start_pos + Vector3(index.x+1, index.y+1, index.z) * block_size;
		t = Vector3(index.x+1, index.y+1, index.z) / size;
		v[6] = { p,t };
		p = start_pos + Vector3(index.x+1, index.y+1, index.z + 1) * block_size;
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
};