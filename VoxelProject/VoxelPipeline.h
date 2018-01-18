#pragma once
#include "D3DSystem.h"

class VoxelPipeline
{
public:
	VoxelPipeline(shared_ptr<D3DSystem> d3dSyst);
	~VoxelPipeline();
private:
	shared_ptr<D3DSystem> m_d3dSyst;
	ComPtr<ID3D12RootSignature> m_meshRootSignature;
	ComPtr<ID3D12PipelineState> m_meshPipelineState;
	RenderingCB m_renderingCB;
	ComPtr<ID3D12Resource> m_constantBufferUploadHeaps[FRAMEBUFFERCOUNT];
	UINT8* m_cbvGPUAddress[FRAMEBUFFERCOUNT];
	D3D12_VIEWPORT m_viewport;
	D3D12_RECT m_scissorRect;
};

