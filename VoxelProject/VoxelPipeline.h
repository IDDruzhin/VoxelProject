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
		RENDER_TEXTURE_UAV = 0,
		BACK_COORD_TEXTURE_UAV = 1,
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

typedef
enum INTERPOLATION_MODE
{
	INTERPOLATION_MODE_NONE = 0,
	INTERPOLATION_MODE_TRILINEAR = 1
} 	INTERPOLATION_MODE;

	VoxelPipeline(shared_ptr<D3DSystem> d3dSyst);
	~VoxelPipeline();
	void RenderObject(VoxelObject* voxObj, Camera* camera, int selectedBone);
	ComPtr<ID3D12Resource> RegisterBlocksInfo(vector<BlockInfo>& blocksInfo);
	ComPtr<ID3D12Resource> RegisterVoxels(vector<Voxel>& voxels);
	ComPtr<ID3D12Resource> RegisterPalette(vector<uchar4>& palette);
	ComPtr<ID3D12Resource> RegisterSegmentsOpacity(vector<float>& segmentsOpacity);
	void SetSegmentsOpacity(vector<float>& segmentsOpacity, ComPtr<ID3D12Resource>& segmentsOpacityRes);
	void ComputeDetectBlocks(int voxelsCount, int3 dim, int blockSize, int3 dimBlocks, int3 min, int3 max, vector<BlockInfo>& blocksInfo, ComPtr<ID3D12Resource> blocksInfoRes);
	void RegisterBlocks(int overlap, int3 dimBlocks, int blockSize, vector<BlockInfo>& blocksInfo, ComPtr<ID3D12Resource>& blocksRes, vector<ComPtr<ID3D12Resource>>& texturesRes, ComPtr<ID3D12Resource>& blocksIndexesRes, vector<BlockPositionInfo>& blocksPosInfo, vector<BlockPriorityInfo>& blocksPriorInfo);
	void ComputeFillBlocks(int voxelsCount, int texturesCount, int3 dim, int blockSize, int3 dimBlocks, int3 min, int3 max, int overlap, vector<ComPtr<ID3D12Resource>>& texturesRes);
	void SetStepSize(float voxelSize, float ratio = 1.0f);
	void SetInterpolationMode(INTERPOLATION_MODE mode);
	void SetBlocksVisiblity(bool isVisible);
	void SetBonesVisiblity(bool isVisible);
private:
	shared_ptr<D3DSystem> m_d3dSyst;
	ComPtr<ID3D12RootSignature> m_renderRootSignature;
	ComPtr<ID3D12PipelineState> m_backFacesPipelineState;
	ComPtr<ID3D12PipelineState> m_rayCastingPipelineState;
	ComPtr<ID3D12PipelineState> m_rayCastingTrilinearPipelineState;
	ComPtr<ID3D12PipelineState> m_blocksRenderPipelineState;
	RenderingCB m_renderingCB;
	ComPtr<ID3D12Resource> m_constantBufferUploadHeaps[FRAMEBUFFERCOUNT];
	UINT8* m_cbvGPUAddress[FRAMEBUFFERCOUNT];
	D3D12_VIEWPORT m_viewport;
	D3D12_RECT m_scissorRect;

	ComPtr<ID3D12Resource> m_opacityUploadBuffer;

	ComPtr<ID3D12Resource> m_blockIndexBuffer;
	D3D12_INDEX_BUFFER_VIEW m_blockIndexBufferView;

	ComPtr<ID3D12Resource> m_boneVertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW m_boneVertexBufferView;

	ComPtr<ID3D12Resource> m_backCoordTexture;
	ComPtr<ID3D12Resource> m_renderTexture;
	ComPtr<ID3D12DescriptorHeap> m_srvUavHeapRender;  ///Cpu read only heap
	ComPtr<ID3D12DescriptorHeap> m_rtvHeapRender;   ///Cpu read/write heap
	UINT m_srvUavDescriptorSize;

	ComPtr<ID3D12RootSignature> m_blocksComputeRootSignature;
	ComPtr<ID3D12PipelineState> m_blocksDetectionPipelineState;
	ComPtr<ID3D12PipelineState> m_blocksFillingPipelineState;
	ComPtr<ID3D12DescriptorHeap> m_blocksComputeSrvUavHeap;

	ComPtr<ID3D12RootSignature> m_bonesRenderRootSignature;
	ComPtr<ID3D12PipelineState> m_bonesRenderPipelineState;
	ComPtr<ID3D12PipelineState> m_bonesEdgesRenderPipelineState;
	RenderBonesCB m_bonesRenderingCB;

	float m_background[4];
	bool m_renderVoxels;
	bool m_renderBlocks;
	bool m_renderBones;
	ComPtr<ID3D12PipelineState> m_selectedRCPipelineState;
	
};
