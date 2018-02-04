#pragma once
#include "D3DSystem.h"
#include "VoxelObject.h"
#include "Camera.h"

class VoxelObject;

class VoxelPipeline
{
public:
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
