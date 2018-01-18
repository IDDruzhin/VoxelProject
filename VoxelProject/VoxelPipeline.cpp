#include "stdafx.h"
#include "VoxelPipeline.h"


VoxelPipeline::VoxelPipeline(shared_ptr<D3DSystem> d3dSyst)
{
	m_d3dSyst = d3dSyst;
	//Viewport
	m_viewport.TopLeftX = 0;
	m_viewport.TopLeftY = 0;
	m_viewport.Width = m_d3dSyst->GetSwapChainDesc().BufferDesc.Width;
	m_viewport.Height = m_d3dSyst->GetSwapChainDesc().BufferDesc.Height;
	m_viewport.MinDepth = 0.0f;
	m_viewport.MaxDepth = 1.0f;

	//ScissorRect
	m_scissorRect.left = 0;
	m_scissorRect.top = 0;
	m_scissorRect.right = m_d3dSyst->GetSwapChainDesc().BufferDesc.Width;
	m_scissorRect.bottom = m_d3dSyst->GetSwapChainDesc().BufferDesc.Height;

	/// Mesh rendering pipeline
	{
		CD3DX12_ROOT_PARAMETER1 rootParameters[1];
		rootParameters[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_VERTEX);

		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
		rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		ID3DBlob *signature;
		ID3DBlob *error;
		ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, m_d3dSyst->GetFeatureData().HighestVersion, &signature, &error));
		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_meshRootSignature)));

		ComPtr<ID3DBlob> vertexShader;
		ComPtr<ID3DBlob> pixelShader;
		UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
		ThrowIfFailed(D3DCompileFromFile(L"MeshVS", nullptr, nullptr, "main", "vs_5_1", compileFlags, 0, &vertexShader, nullptr));
		ThrowIfFailed(D3DCompileFromFile(L"MeshPS", nullptr, nullptr, "main", "ps_5_1", compileFlags, 0, &pixelShader, nullptr));

		D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{"TEXCOORD",0,DXGI_FORMAT_R32G32B32_FLOAT ,0,12,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};

		CD3DX12_BLEND_DESC blendDesc(D3D12_DEFAULT);
		CD3DX12_DEPTH_STENCIL_DESC depthStencilDesc(D3D12_DEFAULT);
		CD3DX12_RASTERIZER_DESC rasterizerDesc(D3D12_DEFAULT);
		rasterizerDesc.FrontCounterClockwise = FALSE;

		// Describe and create the graphics pipeline state object (PSO).
		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
		psoDesc.pRootSignature = m_meshRootSignature.Get();
		psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
		psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
		psoDesc.RasterizerState = rasterizerDesc;
		psoDesc.BlendState = blendDesc;
		psoDesc.DepthStencilState = depthStencilDesc;
		psoDesc.SampleMask = UINT_MAX;
		psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		psoDesc.NumRenderTargets = 1;
		psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
		//psoDesc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
		psoDesc.SampleDesc.Count = 1;

		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_meshPipelineState)));
	}
}

VoxelPipeline::~VoxelPipeline()
{
}
