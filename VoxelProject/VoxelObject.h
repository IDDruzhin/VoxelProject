#pragma once
#include "Object.h"
#include "Structures.h"
#include "VoxelPipeline.h"

class VoxelPipeline;

class VoxelObject :
	public Object
{
public:
	VoxelObject(VoxelPipeline* voxPipeline);
	~VoxelObject();
	ID3D12Resource* GetBlocksRes();
	D3D12_VERTEX_BUFFER_VIEW GetBlocksVertexBufferView();
	int GetBlocksCount();
private:
	Vector3 m_dim;
	Vector3 m_size;
	int m_blockDim;
	float m_blockSize;
	Vector3 m_startPos;
	vector<Voxel> voxels;
	vector<Block> m_blocks;
	vector<BlockInfo> m_blocksInfo;
	vector<Vector3> m_palette;
	vector<float> m_segmentsOpacity;
	ComPtr<ID3D12Resource> m_tex3DRes;
	ComPtr<ID3D12Resource> m_blocksRes;
	D3D12_VERTEX_BUFFER_VIEW m_blocksBufferView;
	ComPtr<ID3D12Resource> m_blocksInfoRes;
};
