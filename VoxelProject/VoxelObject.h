#pragma once
#include "Object.h"
#include "Structures.h"
class VoxelObject :
	public Object
{
public:
	VoxelObject();
	~VoxelObject();
private:
	int m_width;
	int m_height;
	int m_depth;
	float m_blockSize;
	vector<Voxel> voxels;
	vector<Block> m_blocks;
	vector<BlockInfo> m_blocksInfo;
	vector<Vector3> m_palette;
	vector<float> m_segmentsOpacity;
	ComPtr<ID3D12Resource> m_tex3DRes;
	ComPtr<ID3D12Resource> m_blocksRes;
	ComPtr<ID3D12Resource> m_blocksInfoRes;
};

