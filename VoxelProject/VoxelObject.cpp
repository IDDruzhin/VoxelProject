#include "stdafx.h"
#include "VoxelObject.h"


VoxelObject::VoxelObject()
{
	m_width = 608;
	m_height = 346;
	m_depth = 1702;
	Vector3 size(m_width, m_height, m_depth);
	Vector3 startPos = size / max(max(m_width, m_height), m_depth);
	startPos -= startPos / 2.0f;
	float blockDim = 32.0f;
	m_blockSize = blockDim/ max(max(m_width, m_height), m_depth);
	int x_dim = ceil(m_width/blockDim);
	int y_dim = ceil(m_height / blockDim);
	int z_dim = ceil(m_depth / blockDim);
	for (int i = 0; i < x_dim; i++)
	{
		for (int j = 0; j < y_dim; j++)
		{
			for (int k = 0; k < z_dim; k++)
			{
				m_blocks.emplace_back(size, Vector3(i, j, k), startPos, m_blockSize);
				m_blocksInfo.emplace_back(Vector3(i, j, k),blockDim);
			}
		}
	}
}


VoxelObject::~VoxelObject()
{
}
