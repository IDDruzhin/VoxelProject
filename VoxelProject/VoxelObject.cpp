#include "stdafx.h"
#include "VoxelObject.h"


VoxelObject::VoxelObject(VoxelPipeline* voxPipeline)
{
	m_dim.x = 608;
	m_dim.y = 346;
	m_dim.z = 1702;
	float maxSide = max(max(m_dim.x, m_dim.y), m_dim.z);
	m_s = Vector3(1 / maxSide);
	m_t = -m_dim / (2.0f*maxSide);
	m_size = m_dim / maxSide;
	m_startPos = -m_size/2.0f;
	m_blockDim = 32;
	//m_blockDim = 64;
	m_blockSize = m_blockDim/maxSide;
	int x_blockDim = ceil(m_dim.x / m_blockDim);
	int y_blockDim = ceil(m_dim.y / m_blockDim);
	int z_blockDim = ceil(m_dim.z / m_blockDim);
	Vector3 blockIndex(0, 0, 0);
	for (blockIndex.x=0; blockIndex.x < x_blockDim; blockIndex.x++)
	{
		for (blockIndex.y=0; blockIndex.y < y_blockDim; blockIndex.y++)
		{
			for (blockIndex.z=0; blockIndex.z < z_blockDim; blockIndex.z++)
			{
				m_blocks.emplace_back(m_dim, blockIndex, m_startPos, m_blockDim, m_blockSize);
				m_blocksInfo.emplace_back(blockIndex, m_blockDim);
			}
		}
	}
	m_blocksRes = voxPipeline->CreateBlocksViews(&m_blocks[0], m_blocks.size());
	m_blocksBufferView.BufferLocation = m_blocksRes->GetGPUVirtualAddress();
	m_blocksBufferView.StrideInBytes = sizeof(Vertex);
	m_blocksBufferView.SizeInBytes = sizeof(Vertex)*sizeof(Block)*m_blocks.size();
}


VoxelObject::~VoxelObject()
{
}

ID3D12Resource * VoxelObject::GetBlocksRes()
{
	return m_blocksRes.Get();
}

D3D12_VERTEX_BUFFER_VIEW VoxelObject::GetBlocksVertexBufferView()
{
	return m_blocksBufferView;
}

int VoxelObject::GetBlocksCount()
{
	return m_blocks.size();
}

