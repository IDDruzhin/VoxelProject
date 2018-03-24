#pragma once
#include "Object.h"
#include "Structures.h"
#include "VoxelPipeline.h"
#include "Skeleton.h"
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
	void CreateFromSlices(string path);
	void SaveBin(string path, string name);
	void LoadBin(string path);
	void BlocksDecomposition(VoxelPipeline* voxPipeline, int blockSize, int overlay = 0, int3 min = { 0,0,0 }, int3 max = { 0,0,0 });
	vector<BlockPriorityInfo> CalculatePriorities(Vector3 cameraPos);
	D3D12_VERTEX_BUFFER_VIEW GetBlocksVBV();
	float GetVoxelSize();
	vector<string> GetSegmentsNames();
	void SetSegmentsOpacity(VoxelPipeline* voxPipeline, vector<float> &segmentsOpacity);
	vector<float> GetSegmentsOpacity();

	void CopySkeletonMatricesForDraw(Matrix* dst);
	int GetBonesCount();
	void SetSkeletonMatricesForDraw(Matrix viewProj);
private:
	string m_name;
	int3 m_dim;
	vector<Voxel> m_voxels;
	vector<uchar4> m_palette;
	vector<string> m_segmentationTableNames;
	vector<float> m_segmentsOpacity;
	int m_blockSize;

	Skeleton m_skeleton;
	
	vector<ComPtr<ID3D12Resource>> m_texturesRes;
	vector<BlockPriorityInfo> m_blocksPriorInfo;
	vector<BlockPositionInfo> m_blocksPosInfo;
	ComPtr<ID3D12Resource> m_blocksRes;
	D3D12_VERTEX_BUFFER_VIEW m_blocksBufferView;
	ComPtr<ID3D12Resource> m_paletteRes;
	ComPtr<ID3D12Resource> m_segmentsOpacityRes;

};

