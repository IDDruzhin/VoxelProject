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

cbuffer ComputeBlocksCB : register(b0)
{
	int4 min;
	int4 max;
	int4 dim;
	int4 dimBlocks;
	int voxelsCount;
	int blockSize;
	int computeBlocksCount;
	int overlap;
};

StructuredBuffer<Voxel> voxels : register(t0);
//RWStructuredBuffer<BlockInfo> blocksInfo : register(u1);
RWStructuredBuffer<BlockInfo> blocksInfo : register(u0);
//RWStructuredBuffer<BlockInfo> blocksInfo;

[numthreads(blocksize_x, blocksize_y, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	uint size;
	uint offset;
	uint index = DTid.y*computeBlocksCount*blocksize_x + DTid.x;
	voxels.GetDimensions(size, offset);
	//if (index < voxelsCount)
	if (index < size)
	{
		int3 cur;
		int tmp = voxels[index].index % (dim.x*dim.y);
		cur.z = voxels[index].index / (dim.x*dim.y);
		cur.y = tmp / dim.x;
		cur.x = tmp % dim.x;
		if (cur.x >= min.x && cur.x <= max.x && cur.y >= min.y && cur.y <= max.y && cur.z >= min.z && cur.z <= max.z)
		{
			int blockIndex = ((cur.x - min.x) / blockSize) + ((cur.y - min.y) / blockSize) * dimBlocks.x + ((cur.z - min.z) / blockSize) * dimBlocks.x * dimBlocks.y;
			//int blockIndex = ((cur.x - min.x) / blockSize) + ((cur.y - min.y) / blockSize) * blockSize + ((cur.z - min.z) / blockSize) * blockSize * blockSize;
			//int blockIndex = ((cur.x) / 64) + ((cur.y) / 64) * 64 + ((cur.z) / 64) * 64 * 64;
			//int blockIndex = ((cur.x) / 64) + ((cur.y) / blockSize) * blockSize + ((cur.z) / blockSize) * blockSize * blockSize;
			
			InterlockedMin(blocksInfo[blockIndex].min.x, cur.x);
			InterlockedMin(blocksInfo[blockIndex].min.y, cur.y);
			InterlockedMin(blocksInfo[blockIndex].min.z, cur.z);
			InterlockedMax(blocksInfo[blockIndex].max.x, cur.x);
			InterlockedMax(blocksInfo[blockIndex].max.y, cur.y);
			InterlockedMax(blocksInfo[blockIndex].max.z, cur.z);
			
			/*
			int3 tstMin = { 1,2,3 };
			blocksInfo[blockIndex].min = tstMin;
			int3 tstMax = { 10,20,30 };
			blocksInfo[blockIndex].max = tstMax;
			*/
			//InterlockedAdd(blocksInfo[0].min.x, 1);
			
		}
	}	
	/*
	blocksInfo.GetDimensions(size, offset);
	if (index < size)
	{
		blocksInfo[index].min.x = blockSize;
		blocksInfo[index].max.x = computeBlocksCount;
	}
	*/
}