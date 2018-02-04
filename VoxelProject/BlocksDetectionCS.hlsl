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

struct ComputeBlocksCB : register(b0)
{
	int3 min;
	int3 max;
	int3 dim;
	int3 dimBlocks;
};

StructuredBuffer<Voxel> voxels : register(t0);
RWStructuredBuffer<BlockInfo> blocksInfo : register(u1);

[numthreads(blocksize_x, blocksize_y, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
}