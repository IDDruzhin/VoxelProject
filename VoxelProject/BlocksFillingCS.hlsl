#define blocksize_x 32
#define blocksize_y 32

struct Voxel
{
	uint index;
	uint info;
	//1 byte - color
	//2 byte - segment
	//3 byte - padding
	//4 byte - padding
};

struct BlockInfo
{
	int3 min;
	int3 max;
};

struct Vertex
{
	float3 pos;
	float3 tex;
};

struct Block
{
	Vertex v[8];
};

cbuffer ComputeBlocksCB : register(b0)
{
	int4 min;
	int4 max;
	int4 dim;
	int4 dimBlocks;
	int blockSize;
	int computeBlocksCount;
	int overlap;
};

StructuredBuffer<Voxel> voxels : register(t0);
StructuredBuffer<BlockInfo> blocksInfo : register(t1);
StructuredBuffer<int> blocksIndexes : register(t2);
RWTexture3D<int2> textures[] : register(u2);

[numthreads(blocksize_x, blocksize_y, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
}