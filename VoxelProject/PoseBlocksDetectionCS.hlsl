#define blocksize_x 32
#define blocksize_y 32

struct Voxel
{
	uint index;
	uint info;
	//1 byte - color
	//2 byte - segment
	//3 byte - bone00 index
	//4 byte - bone01 index
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
	float4x4 bones[256];
};

StructuredBuffer<Voxel> voxels : register(t0);
StructuredBuffer<float> bonesWeights00 : register(t3);
StructuredBuffer<float> bonesWeights01 : register(t4);
StructuredBuffer<uint> additionalBones : register(t5);
RWStructuredBuffer<BlockInfo> blocksInfo : register(u0);

[numthreads(blocksize_x, blocksize_y, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	uint index = DTid.y*computeBlocksCount*blocksize_x + DTid.x;
	if (index < voxelsCount)
	{
		Voxel voxel = voxels[index];
		uint bone00 = ((voxel.info >> 16) & 255);
		if (bone00 == 0)
		{
			return;
		}
		uint bone01 = ((voxel.info >> 24) & 255);
		uint bone02 = ((additionalBones[index / 4] >> ((index % 4) * 8)) & 255);
		float3 pos;
		int tmp = voxel.index % (dim.x*dim.y);
		pos.z = voxel.index / (dim.x*dim.y);
		pos.y = tmp / dim.x;
		pos.x = tmp % dim.x;
		//float4x4 poseMatrix = lerp(bones[bone01], bones[bone00], bonesWeights[index]);
		float4x4 poseMatrix = mul(bones[bone00], bonesWeights00[index]) + mul(bones[bone01], bonesWeights01[index]) + mul(bones[bone02], (1.0f - bonesWeights00[index] - bonesWeights01[index]));
		for (int i = 0; i <= 1; i++)
		{
			for (int j = 0; j <= 1; j++)
			{
				for (int k = 0; k <= 1; k++)
				{
					float4 cur = float4(pos.x + i, pos.y + j, pos.z + k, 1.0f);
					cur = mul(cur, poseMatrix);
					if (cur.x >= min.x && cur.x <= max.x && cur.y >= min.y && cur.y <= max.y && cur.z >= min.z && cur.z <= max.z)
					{
						int blockIndex = (((int)cur.x - min.x) / blockSize) + (((int)cur.y - min.y) / blockSize) * dimBlocks.x + (((int)cur.z - min.z) / blockSize) * dimBlocks.x * dimBlocks.y;
						InterlockedMin(blocksInfo[blockIndex].min.x, cur.x);
						InterlockedMin(blocksInfo[blockIndex].min.y, cur.y);
						InterlockedMin(blocksInfo[blockIndex].min.z, cur.z);
						InterlockedMax(blocksInfo[blockIndex].max.x, cur.x);
						InterlockedMax(blocksInfo[blockIndex].max.y, cur.y);
						InterlockedMax(blocksInfo[blockIndex].max.z, cur.z);
					}
				}
			}
		}
	}
}