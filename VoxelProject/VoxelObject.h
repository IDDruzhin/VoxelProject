#pragma once
#include "Object.h"
#include "Structures.h"
#include "VoxelPipeline.h"
#include <sstream>


#include "CudaFunctions.cuh"


class VoxelPipeline;

class VoxelObject :
	public Object
{
public:
typedef
	enum LOADING_MODE
{
	LOADING_MODE_SLICES = 0,
	LOADING_MODE_BIN = 1
} 	LOADING_MODE;
	VoxelObject(VoxelPipeline* voxPipeline);
	VoxelObject(string path, LOADING_MODE loadingMode, VoxelPipeline* voxPipeline);
	~VoxelObject();
	ID3D12Resource* GetBlocksRes();
	D3D12_VERTEX_BUFFER_VIEW GetBlocksVertexBufferView();
	int GetBlocksCount();
	void CreateFromSlices(string path, VoxelPipeline* voxPipeline);
	void SaveBin(string path, string name);
	void LoadBin(string path);
private:
	vector<Voxel> m_voxels;
	vector<uchar4> m_palette;
	//vector<SegmentData> segmentationTable;
	vector<string> m_segmentationTableNames;

	string m_name;
	uint3 m_dim;
	Vector3 m_size;
	int m_blockDim;
	float m_blockSize;
	Vector3 m_startPos;
	vector<Block> m_blocks;
	vector<BlockInfo> m_blocksInfo;
	//vector<Vector3> m_palette;
	vector<float> m_segmentsOpacity;
	ComPtr<ID3D12Resource> m_tex3DRes;
	ComPtr<ID3D12Resource> m_blocksRes;
	D3D12_VERTEX_BUFFER_VIEW m_blocksBufferView;
	ComPtr<ID3D12Resource> m_blocksInfoRes;
};

