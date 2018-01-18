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
		ThrowIfFailed(D3DCompileFromFile(L"MeshVS.hlsl", nullptr, nullptr, "main", "vs_5_1", compileFlags, 0, &vertexShader, nullptr));
		ThrowIfFailed(D3DCompileFromFile(L"MeshPS.hlsl", nullptr, nullptr, "main", "ps_5_1", compileFlags, 0, &pixelShader, nullptr));

		D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{"TEXCOORD",0,DXGI_FORMAT_R32G32B32_FLOAT ,0,12,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};

		CD3DX12_BLEND_DESC blendDesc(D3D12_DEFAULT);
		CD3DX12_DEPTH_STENCIL_DESC depthStencilDesc(D3D12_DEFAULT);
		CD3DX12_RASTERIZER_DESC rasterizerDesc(D3D12_DEFAULT);
		//rasterizerDesc.FrontCounterClockwise = FALSE;
		//rasterizerDesc.CullMode = D3D12_CULL_MODE_NONE;
		//rasterizerDesc.CullMode = D3D12_CULL_MODE_FRONT;
		rasterizerDesc.CullMode = D3D12_CULL_MODE_BACK;	

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

	///Constant buffer
	ZeroMemory(&m_renderingCB, sizeof(m_renderingCB));
	for (int i = 0; i < FRAMEBUFFERCOUNT; ++i)
	{
		ThrowIfFailed(d3dSyst->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(1024 * 64), //Size. Must be a multiple of 64KB for constant buffers
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_constantBufferUploadHeaps[i])));
		m_constantBufferUploadHeaps[i]->SetName(L"Constant Buffer Upload Resource Heap");
		CD3DX12_RANGE readRange(0, 0);    //No reading on CPU
		ThrowIfFailed(m_constantBufferUploadHeaps[i]->Map(0, &readRange, reinterpret_cast<void**>(&m_cbvGPUAddress[i])));
	}
	int blockIndexes[] =
	{
		0,6,4,
		0,2,6,
		0,3,2,
		0,1,3,
		2,7,6,
		2,3,7,
		4,6,7,
		4,7,5,
		0,4,5,
		0,5,1,
		1,5,7,
		1,7,3
	};
	m_blockIndexBuffer = m_d3dSyst->CreateIndexBuffer(&blockIndexes[0], sizeof(blockIndexes), L"Blocks index buffer");
	m_blockIndexBufferView.BufferLocation = m_blockIndexBuffer->GetGPUVirtualAddress();
	m_blockIndexBufferView.Format = DXGI_FORMAT_R32_UINT;
	m_blockIndexBufferView.SizeInBytes = sizeof(blockIndexes);
}

VoxelPipeline::~VoxelPipeline()
{
}

void VoxelPipeline::RenderObject(VoxelObject * voxObj, Camera* camera)
{
	if (voxObj != nullptr)
	{
		m_d3dSyst->Reset();
		int frameIndex = m_d3dSyst->GetFrameIndex();
		m_d3dSyst->UpdatePipelineAndClear(Vector3(0, 0, 0));
		ComPtr<ID3D12GraphicsCommandList> commandList = m_d3dSyst->GetCommandList();
		m_renderingCB.worldViewProj = (voxObj->GetWorld()*camera->GetView()*camera->GetProjection()).Transpose();
		commandList->SetPipelineState(m_meshPipelineState.Get());
		commandList->SetGraphicsRootSignature(m_meshRootSignature.Get());
		commandList->RSSetViewports(1, &m_viewport);
		commandList->RSSetScissorRects(1, &m_scissorRect);
		commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		memcpy(m_cbvGPUAddress[frameIndex], &m_renderingCB, sizeof(m_renderingCB));
		commandList->SetGraphicsRootConstantBufferView(0, m_constantBufferUploadHeaps[frameIndex]->GetGPUVirtualAddress());
		commandList->IASetVertexBuffers(0, 1, &(voxObj->GetBlocksVertexBufferView()));
		//commandList->DrawInstanced(3, 1, 0, 0);
		commandList->IASetIndexBuffer(&m_blockIndexBufferView);
		//commandList->DrawIndexedInstanced(36, 1, 0, 0, 0);
		//commandList->DrawIndexedInstanced(36, 1, 0, 0, 36*100);
		//commandList->DrawIndexedInstanced(36, voxObj->GetBlocksCount(), 0, 0, 8);
		//commandList->DrawIndexedInstanced(36, voxObj->GetBlocksCount(), 0, 0, 8);
		for (int i = 0; i < voxObj->GetBlocksCount(); i++)
		{
			commandList->DrawIndexedInstanced(36, 1, 0, 8*i, 0);
		}
		m_d3dSyst->ExecuteGraphics();
		m_d3dSyst->PresentSimple();
	}
}

