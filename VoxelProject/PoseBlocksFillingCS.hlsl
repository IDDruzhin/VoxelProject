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
	int voxelsCount;
	int blockSize;
	int computeBlocksCount;
	int overlap;
	float4x4 bones[256];
};

StructuredBuffer<Voxel> voxels : register(t0);
StructuredBuffer<BlockInfo> blocksInfo : register(t1);
StructuredBuffer<int> blocksIndexes : register(t2);
StructuredBuffer<float> bonesWeights00 : register(t3);
StructuredBuffer<float> bonesWeights01 : register(t4);
StructuredBuffer<uint> additionalBones : register(t5);
RWTexture3D<uint2> textures[] : register(u1);

void FillElement(int3 pos, int color, int segment, int3 block3dIndex)
{
	int blockIndex = block3dIndex.x + block3dIndex.y * dimBlocks.x + block3dIndex.z * dimBlocks.x * dimBlocks.y;
	int textureIndex = blocksIndexes[blockIndex];
	if (textureIndex > -1)
	{
		BlockInfo blockInfo = blocksInfo[blockIndex];
		if ((pos.x >= (blockInfo.min.x - overlap)) && (pos.x <= (blockInfo.max.x + overlap)) && (pos.y >= (blockInfo.min.y - overlap)) && (pos.y <= (blockInfo.max.y + overlap)) && (pos.z >= (blockInfo.min.z - overlap)) && (pos.z <= (blockInfo.max.z + overlap)))
		{
			uint3 texturePos = uint3(pos.x - blockInfo.min.x + overlap, pos.y - blockInfo.min.y + overlap, pos.z - blockInfo.min.z + overlap);
			uint2 element = uint2(color, segment);
			textures[textureIndex][texturePos] = element;
		}
	}
}

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
		int color = (voxel.info & 255);
		int segment = ((voxel.info >> 8) & 255);
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
						int3 block3dIndex = int3(((int)cur.x - min.x) / blockSize, ((int)cur.y - min.y) / blockSize, ((int)cur.z - min.z) / blockSize);
						if (overlap == 0)
						{
							FillElement(cur, color, segment, block3dIndex);
						}
						else
						{
							for (int i = -1; i <= 1; i += 1)
							{
								for (int j = -1; j <= 1; j += 1)
								{
									for (int k = -1; k <= 1; k += 1)
									{
										int3 neighborBlock3dIndex = int3(block3dIndex.x + i, block3dIndex.y + j, block3dIndex.z + k);
										if ((neighborBlock3dIndex.x > -1) && (neighborBlock3dIndex.x < dimBlocks.x) && (neighborBlock3dIndex.y > -1) && (neighborBlock3dIndex.y < dimBlocks.y) && (neighborBlock3dIndex.z > -1) && (neighborBlock3dIndex.z < dimBlocks.z))
										{
											FillElement(cur, color, segment, neighborBlock3dIndex);
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}