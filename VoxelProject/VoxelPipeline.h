#pragma once
#include "D3DSystem.h"
#include "VoxelObject.h"
#include "Camera.h"

class VoxelObject;

class VoxelPipeline
{
public:
typedef
	enum GRAPHICS_DESCRIPTORS
	{
		POSITION_TEXTURE_UAV = 0,
		RENDER_TEXTURE_UAV = 1,
		PALETTE_SRV = 2,
		SEGMENTS_OPACITY_SRV = 3,
		TEXTURES_3D_SRV_ARRAY = 4
	} 	GRAPHICS_DESCRIPTORS;
typedef
	enum COMPUTE_DESCRIPTORS
	{
		VOXELS_SRV = 0,
		BLOCKS_INFO_SRV = 1,
		BLOCKS_INDEXES_SRV = 2,
		BLOCKS_INFO_UAV = 3,
		TEXTURES_3D_UAV_ARRAY = 4
	} 	COMPUTE_DESCRIPTORS;

	VoxelPipeline(shared_ptr<D3DSystem> d3dSyst);
	~VoxelPipeline();
	void RenderObject(VoxelObject* voxObj, Camera* camera);
	/*
	template<typename T>
	ComPtr<ID3D12Resource> Create3dTextureViews(T* data, int elementsCount);
	template<typename T>
	ComPtr<ID3D12Resource> CreateBlocksViews(T* data, int elementsCount);
	*/
	ComPtr<ID3D12Resource> RegisterBlocksInfo(vector<BlockInfo>& blocksInfo);
	ComPtr<ID3D12Resource> RegisterVoxels(vector<Voxel>& voxels);
	void ComputeDetectBlocks(int voxelsCount, int3 dim, int blockSize, int3 dimBlocks, int3 min, int3 max, vector<BlockInfo>& blocksInfo, ComPtr<ID3D12Resource> blocksInfoRes);
	void RegisterBlocks(int overlap, int3 dimBlocks, vector<BlockInfo>& blocksInfo, ComPtr<ID3D12Resource>& blocksRes, vector<ComPtr<ID3D12Resource>>& texturesRes, vector<int>& blocksIndexes, ComPtr<ID3D12Resource>& blocksIndexesRes, vector<int3>& blocks3dIndexes, vector<Vector3>& blocksPositions);
	void ComputeFillBlocks(int voxelsCount, int3 dim, int blockSize, int3 dimBlocks, int3 min, int3 max, int overlap, vector<ComPtr<ID3D12Resource>>& texturesRes);
private:
	shared_ptr<D3DSystem> m_d3dSyst;
	ComPtr<ID3D12RootSignature> m_meshRootSignature;
	ComPtr<ID3D12PipelineState> m_meshPipelineState;
	RenderingCB m_renderingCB;
	ComPtr<ID3D12Resource> m_constantBufferUploadHeaps[FRAMEBUFFERCOUNT];
	UINT8* m_cbvGPUAddress[FRAMEBUFFERCOUNT];
	D3D12_VIEWPORT m_viewport;
	D3D12_RECT m_scissorRect;

	ComPtr<ID3D12Resource> m_blockIndexBuffer;
	D3D12_INDEX_BUFFER_VIEW m_blockIndexBufferView;

	ComPtr<ID3D12Resource> m_renderTexture;
	ComPtr<ID3D12DescriptorHeap> m_srvUavHeapRender;  ///Cpu read only heap
	ComPtr<ID3D12DescriptorHeap> m_rtvHeapRender;   ///Cpu read/write heap
	UINT m_srvUavDescriptorSize;

	//ComPtr<ID3D12Resource> m_constantBufferUploadHeapCompute;
	ComPtr<ID3D12RootSignature> m_blocksComputeRootSignature;
	ComPtr<ID3D12PipelineState> m_blocksDetectionPipelineState;
	ComPtr<ID3D12PipelineState> m_blocksFillingPipelineState;
	ComPtr<ID3D12DescriptorHeap> m_blocksComputeSrvUavHeap;
	
};

/*
template<typename T>
inline ComPtr<ID3D12Resource> VoxelPipeline::Create3dTextureViews(T * data, int elementsCount)
{
	//ComPtr<ID3D12Resource> buffer = m_d3dSyst->CreateVertexBuffer(data, elementsCount,L"")
	return ComPtr<ID3D12Resource>();
}

template<typename T>
inline ComPtr<ID3D12Resource> VoxelPipeline::CreateBlocksViews(T * data, int elementsCount)
{
	ComPtr<ID3D12Resource> buffer = m_d3dSyst->CreateVertexBuffer(data, elementsCount*sizeof(T), L"Blocks vertex buffer");
	return buffer;
}
*/
