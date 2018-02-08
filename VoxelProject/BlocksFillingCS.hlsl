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
	int voxelsCount;
	int blockSize;
	int computeBlocksCount;
	int overlap;
};

StructuredBuffer<Voxel> voxels : register(t0);
StructuredBuffer<BlockInfo> blocksInfo : register(t1);
StructuredBuffer<int> blocksIndexes : register(t2);
//RWStructuredBuffer<BlockInfo> blocksK : register(u1);
//RWTexture3D<uint2> textures[] : register(u1);  //u1 + 1 = first texture
RWTexture3D<uint2> textures[] : register(u1);
//RWTexture3D<uint2> textures[];

void FillElement(int3 pos, int color, int segment, int3 block3dIndex)
{
	int blockIndex = block3dIndex.x + block3dIndex.y * dimBlocks.x + block3dIndex.z * dimBlocks.x * dimBlocks.y;
	int textureIndex = blocksIndexes[blockIndex];
	//textures[0][uint3(0, 0, 0)] = uint2(200, 10);
	if (textureIndex > -1)
	{
		BlockInfo blockInfo = blocksInfo[blockIndex];		
		if ((pos.x >= blockInfo.min.x - overlap) && (pos.x <= blockInfo.max.x + overlap) && (pos.y >= blockInfo.min.y - overlap) && (pos.y <= blockInfo.max.y + overlap) && (pos.z >= blockInfo.min.z - overlap) && (pos.z <= blockInfo.max.z + overlap))
		{
			uint3 texturePos = uint3(pos.x - blockInfo.min.x + overlap, pos.y - blockInfo.min.y + overlap, pos.z - blockInfo.min.z + overlap);
			//textures[textureIndex][texturePos].x = color;
			//textures[textureIndex][texturePos].y = segment;
			uint2 element = uint2(color, segment);
			textures[textureIndex][texturePos] = element;
		}
		//uint3 texturePos = uint3(pos.x - blockInfo.min.x, pos.y - blockInfo.min.y, pos.z - blockInfo.min.z);
		//textures[textureIndex][uint3(0,0,0)] = uint2(200,10);
	}
}

[numthreads(blocksize_x, blocksize_y, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	uint index = DTid.y*computeBlocksCount*blocksize_x + DTid.x;
	//textures[-1][uint3(0, 0, 0)] = uint2(100, 100);
	//textures[uint3(0, 0, 0)] = uint2(200, 10);
	//textures[1][uint3(0, 0, 0)] = uint2(20, 100);
	if (index < voxelsCount)
	{
		int3 cur;
		Voxel voxel = voxels[index];
		int tmp = voxel.index % (dim.x*dim.y);
		cur.z = voxel.index / (dim.x*dim.y);
		cur.y = tmp / dim.x;
		cur.x = tmp % dim.x;
		if (cur.x >= min.x && cur.x <= max.x && cur.y >= min.y && cur.y <= max.y && cur.z >= min.z && cur.z <= max.z)
		{
			int color = (voxel.info & 255);
			int segment = ((voxel.info >> 8) & 255);
			int3 block3dIndex = int3((cur.x - min.x) / blockSize, (cur.y - min.y) / blockSize, (cur.z - min.z) / blockSize);
			FillElement(cur, color, segment, block3dIndex);
			if (overlap != 0)
			{
				for (int i = -1; i <= 1; i += 2)
				{
					for (int j = -1; j <= 1; j += 2)
					{
						for (int k = -1; k <= 1; k += 2)
						{
							int3 neighborBlock3dIndex = int3(block3dIndex.x + i, block3dIndex.y + j, block3dIndex.z + k);
							if ((neighborBlock3dIndex.x > -1) && (neighborBlock3dIndex.y > -1) && (neighborBlock3dIndex.z > -1))
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