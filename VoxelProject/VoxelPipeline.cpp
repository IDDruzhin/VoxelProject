#include "stdafx.h"
#include "VoxelPipeline.h"


VoxelPipeline::VoxelPipeline(shared_ptr<D3DSystem> d3dSyst) : m_renderBlocks(false)
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
		CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 1, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);
		ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, -1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);

		CD3DX12_ROOT_PARAMETER1 rootParameters[3];
		rootParameters[0].InitAsConstantBufferView(1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_ALL);
		rootParameters[1].InitAsDescriptorTable(2, &ranges[0], D3D12_SHADER_VISIBILITY_PIXEL);
		rootParameters[2].InitAsConstants(1, 0, 0, D3D12_SHADER_VISIBILITY_PIXEL);

		CD3DX12_STATIC_SAMPLER_DESC sampler(0, D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_BORDER, D3D12_TEXTURE_ADDRESS_MODE_BORDER, D3D12_TEXTURE_ADDRESS_MODE_BORDER,
			0.0f, 0, D3D12_COMPARISON_FUNC_NEVER, D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK, 0.0f, D3D12_FLOAT32_MAX, D3D12_SHADER_VISIBILITY_PIXEL, 0);

		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
		rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);


		ID3DBlob *signature;
		ID3DBlob *error;
		ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, m_d3dSyst->GetFeatureData().HighestVersion, &signature, &error));
		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_renderRootSignature)));

		ComPtr<ID3DBlob> vertexShader;
		ComPtr<ID3DBlob> pixelShader;
#if defined(_DEBUG)
		D3DReadFileToBlob(L"shaders_debug\\MeshVS.cso", &vertexShader);
		D3DReadFileToBlob(L"shaders_debug\\BackFacesPS.cso", &pixelShader);
#else
		D3DReadFileToBlob(L"shaders\\MeshVS.cso", &vertexShader);
		D3DReadFileToBlob(L"shaders\\BackFacesPS.cso", &pixelShader);
#endif

		D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{"TEXCOORD",0,DXGI_FORMAT_R32G32B32_FLOAT ,0,12,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};

		CD3DX12_BLEND_DESC blendDesc(D3D12_DEFAULT);
		CD3DX12_DEPTH_STENCIL_DESC depthStencilDesc(D3D12_DEFAULT);
		CD3DX12_RASTERIZER_DESC rasterizerDesc(D3D12_DEFAULT);
		rasterizerDesc.CullMode = D3D12_CULL_MODE_FRONT;

		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
		psoDesc.pRootSignature = m_renderRootSignature.Get();
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
		psoDesc.SampleDesc.Count = 1;

		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_backFacesPipelineState)));

#if defined(_DEBUG)
		D3DReadFileToBlob(L"shaders_debug\\RayCastingPS.cso", &pixelShader);
#else
		D3DReadFileToBlob(L"shaders\\RayCastingPS.cso", &pixelShader);
#endif	
		rasterizerDesc.CullMode = D3D12_CULL_MODE_BACK;
		psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
		psoDesc.RasterizerState = rasterizerDesc;
		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_rayCastingPipelineState)));

#if defined(_DEBUG)
		D3DReadFileToBlob(L"shaders_debug\\RayCastingTrilinearPS.cso", &pixelShader);
#else
		D3DReadFileToBlob(L"shaders\\RayCastingTrilinearPS.cso", &pixelShader);
#endif	
		psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_rayCastingTrilinearPipelineState)));

#if defined(_DEBUG)
		D3DReadFileToBlob(L"shaders_debug\\SimpleColorPS.cso", &pixelShader);
#else
		D3DReadFileToBlob(L"shaders\\SimpleColorPS.cso", &pixelShader);
#endif	
		rasterizerDesc.FillMode = D3D12_FILL_MODE_WIREFRAME;
		psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
		psoDesc.RasterizerState = rasterizerDesc;
		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_blocksRenderPipelineState)));
	}

	/// Blocks detection and filling compute pipeline
	{
		D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
		srvUavHeapDesc.NumDescriptors = 10000;
		srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		d3dSyst->GetDevice()->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&m_blocksComputeSrvUavHeap));

		CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE); //t0-t2
		ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, -1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE); //u1-...

		CD3DX12_ROOT_PARAMETER1 rootParameters[2];
		rootParameters[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_ALL);
		rootParameters[1].InitAsDescriptorTable(2, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);

		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
		rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

		ID3DBlob *signature;
		ID3DBlob *error;
		ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, m_d3dSyst->GetFeatureData().HighestVersion, &signature, &error));
		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_blocksComputeRootSignature)));

		ComPtr<ID3DBlob> computeShader;
#if defined(_DEBUG)
		D3DReadFileToBlob(L"shaders_debug\\BlocksDetectionCS.cso", &computeShader);
#else
		D3DReadFileToBlob(L"shaders\\BlocksDetectionCS.cso", &computeShader);
#endif
		D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
		computePsoDesc.pRootSignature = m_blocksComputeRootSignature.Get();
		computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());
		ThrowIfFailed(d3dSyst->GetDevice()->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_blocksDetectionPipelineState)));

#if defined(_DEBUG)
		D3DReadFileToBlob(L"shaders_debug\\BlocksFillingCS.cso", &computeShader);
#else
		D3DReadFileToBlob(L"shaders\\BlocksFillingCS.cso", &computeShader);
#endif
		computePsoDesc.pRootSignature = m_blocksComputeRootSignature.Get();
		computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());
		ThrowIfFailed(d3dSyst->GetDevice()->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_blocksFillingPipelineState)));
	}

	///Constant buffer
	ZeroMemory(&m_renderingCB, sizeof(m_renderingCB));
	for (int i = 0; i < FRAMEBUFFERCOUNT; ++i)
	{
		ThrowIfFailed(d3dSyst->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(1024 * 64), //Size must be a multiple of 64KB for constant buffers
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_constantBufferUploadHeaps[i])));
		m_constantBufferUploadHeaps[i]->SetName(L"Constant buffer upload heap");
		CD3DX12_RANGE readRange(0, 0);    //No reading on CPU
		ThrowIfFailed(m_constantBufferUploadHeaps[i]->Map(0, &readRange, reinterpret_cast<void**>(&m_cbvGPUAddress[i])));
	}

	m_srvUavDescriptorSize = m_d3dSyst->GetDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	///Render texture
	{
		D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
		srvUavHeapDesc.NumDescriptors = 10000;
		srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		m_d3dSyst->GetDevice()->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&m_srvUavHeapRender));

		D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
		rtvHeapDesc.NumDescriptors = 10;
		rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		m_d3dSyst->GetDevice()->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeapRender));

		DXGI_SWAP_CHAIN_DESC swapChainDesc = m_d3dSyst->GetSwapChainDesc();
		CD3DX12_RESOURCE_DESC backTextureDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R32G32B32A32_FLOAT, swapChainDesc.BufferDesc.Width, swapChainDesc.BufferDesc.Height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), //A default heap
			D3D12_HEAP_FLAG_NONE, //No flags
			&backTextureDesc,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			nullptr,
			IID_PPV_ARGS(&m_backCoordTexture)));
		m_backCoordTexture->SetName(L"Back faces texture coordinates texture");

		D3D12_UNORDERED_ACCESS_VIEW_DESC backUavDesc = {};
		backUavDesc.Format = backTextureDesc.Format;
		backUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

		CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandleBack(m_srvUavHeapRender->GetCPUDescriptorHandleForHeapStart(), GRAPHICS_DESCRIPTORS::BACK_COORD_TEXTURE_UAV, m_srvUavDescriptorSize);
		m_d3dSyst->GetDevice()->CreateUnorderedAccessView(m_backCoordTexture.Get(), nullptr, &backUavDesc, uavHandleBack);

		CD3DX12_RESOURCE_DESC renderTextureDesc = CD3DX12_RESOURCE_DESC::Tex2D(swapChainDesc.BufferDesc.Format, swapChainDesc.BufferDesc.Width, swapChainDesc.BufferDesc.Height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
			D3D12_HEAP_FLAG_NONE,
			&renderTextureDesc,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			nullptr,
			IID_PPV_ARGS(&m_renderTexture)));
		m_renderTexture->SetName(L"Render texture");

		D3D12_UNORDERED_ACCESS_VIEW_DESC renderUavDesc = {};
		renderUavDesc.Format = renderTextureDesc.Format;
		renderUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

		CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandleRender(m_srvUavHeapRender->GetCPUDescriptorHandleForHeapStart(), GRAPHICS_DESCRIPTORS::RENDER_TEXTURE_UAV, m_srvUavDescriptorSize);
		m_d3dSyst->GetDevice()->CreateUnorderedAccessView(m_renderTexture.Get(), nullptr, &renderUavDesc, uavHandleRender);

		CD3DX12_CPU_DESCRIPTOR_HANDLE cpuWriteUavHandleRender(m_rtvHeapRender->GetCPUDescriptorHandleForHeapStart(), GRAPHICS_DESCRIPTORS::RENDER_TEXTURE_UAV, m_srvUavDescriptorSize);
		m_d3dSyst->GetDevice()->CreateUnorderedAccessView(m_renderTexture.Get(), nullptr, &renderUavDesc, cpuWriteUavHandleRender);
	}

	///Block indexes
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

	///Bone mesh
	float S = sin(XM_PI / 3);
	vector<Vector3> vertices;
	vertices.push_back(Vector3(0.0f, -0.5f, -S));
	vertices.push_back(Vector3(0.0f, 1.0f, 0.0f));
	vertices.push_back(Vector3(1.0f, 0.0f, 0.0f));
	vertices.push_back(Vector3(0.0f, -0.5f, S));
	vertices.push_back(Vector3(0.0f, -0.5f, -S));
	vertices.push_back(Vector3(0.0f, 1.0f, 0.0f));
	m_boneVertexBuffer = d3dSyst->CreateVertexBuffer(&vertices[0], vertices.size() * sizeof(Vector3), L"Bone mesh vertex buffer");
	m_boneVertexBufferView.BufferLocation = m_boneVertexBuffer->GetGPUVirtualAddress();
	m_boneVertexBufferView.StrideInBytes = sizeof(Vector3);
	m_boneVertexBufferView.SizeInBytes = vertices.size() * sizeof(Vector3);

	m_background[0] = 0.0f;
	m_background[1] = 0.0f;
	m_background[2] = 0.0f;
	m_background[3] = 1.0f;

	m_d3dSyst->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(sizeof(float)*256),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&m_opacityUploadBuffer));
	m_opacityUploadBuffer->SetName(L"Opacity upload buffer");

	SetInterpolationMode(INTERPOLATION_MODE::INTERPOLATION_MODE_NONE);
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
		m_d3dSyst->UpdatePipeline();
		
		ComPtr<ID3D12GraphicsCommandList> commandList = m_d3dSyst->GetCommandList();
		Matrix worldView = voxObj->GetWorld() * camera->GetView();
		m_renderingCB.worldView = worldView.Transpose();
		m_renderingCB.worldViewProj = (worldView * camera->GetProjection()).Transpose();
		
		ID3D12DescriptorHeap* heaps[] = { m_srvUavHeapRender.Get() };
		commandList->SetDescriptorHeaps(_countof(heaps), heaps);

		commandList->SetGraphicsRootSignature(m_renderRootSignature.Get());
		commandList->RSSetViewports(1, &m_viewport);
		commandList->RSSetScissorRects(1, &m_scissorRect);
		commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		memcpy(m_cbvGPUAddress[frameIndex], &m_renderingCB, sizeof(m_renderingCB));
		commandList->SetGraphicsRootConstantBufferView(0, m_constantBufferUploadHeaps[frameIndex]->GetGPUVirtualAddress());
		commandList->SetGraphicsRootDescriptorTable(1, m_srvUavHeapRender->GetGPUDescriptorHandleForHeapStart());

		CD3DX12_GPU_DESCRIPTOR_HANDLE gpuH(m_srvUavHeapRender->GetGPUDescriptorHandleForHeapStart(), GRAPHICS_DESCRIPTORS::RENDER_TEXTURE_UAV, m_srvUavDescriptorSize);
		CD3DX12_CPU_DESCRIPTOR_HANDLE cpuH(m_rtvHeapRender->GetCPUDescriptorHandleForHeapStart(), GRAPHICS_DESCRIPTORS::RENDER_TEXTURE_UAV, m_srvUavDescriptorSize);
		commandList->ClearUnorderedAccessViewFloat(gpuH, cpuH, m_renderTexture.Get(), &m_background[0],0,nullptr);
		
		
		commandList->IASetVertexBuffers(0, 1, &(voxObj->GetBlocksVBV()));
		commandList->IASetIndexBuffer(&m_blockIndexBufferView);

		Matrix invertWorldView = (voxObj->GetWorld()*camera->GetView()).Invert();
		Vector3 cameraPos(0.0f,0.0f,0.0f);
		cameraPos = Vector3::Transform(cameraPos, invertWorldView);
		vector<BlockPriorityInfo> blocksOrder = voxObj->CalculatePriorities(cameraPos);
		int priority = blocksOrder[0].priority;
		int start = 0;
		for (int i = 0; i < blocksOrder.size(); i++)
		{
			if ((priority != blocksOrder[i].priority))
			{
				commandList->SetPipelineState(m_backFacesPipelineState.Get());
				for (int j = start; j < i; j++)
				{
					commandList->DrawIndexedInstanced(36, 1, 0, 8 * blocksOrder[j].blockIndex, 0);
				}
				commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_backCoordTexture.Get()));
				commandList->SetPipelineState(m_selectedRCPipelineState.Get());
				for (int j = start; j < i; j++)
				{
					commandList->SetGraphicsRoot32BitConstant(2, blocksOrder[j].blockIndex, 0);
					commandList->DrawIndexedInstanced(36, 1, 0, 8 * blocksOrder[j].blockIndex, 0);
				}
				commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_renderTexture.Get()));

				priority = blocksOrder[i].priority;
				start = i;
			}
		}
		commandList->SetPipelineState(m_backFacesPipelineState.Get());
		for (int j = start; j < blocksOrder.size(); j++)
		{
			commandList->DrawIndexedInstanced(36, 1, 0, 8 * blocksOrder[j].blockIndex, 0);
		}
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_backCoordTexture.Get()));

		commandList->SetPipelineState(m_selectedRCPipelineState.Get());
		for (int j = start; j < blocksOrder.size(); j++)
		{
			commandList->SetGraphicsRoot32BitConstant(2, blocksOrder[j].blockIndex, 0);
			commandList->DrawIndexedInstanced(36, 1, 0, 8 * blocksOrder[j].blockIndex, 0);
		}
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_renderTexture.Get()));

		commandList->CopyResource(m_d3dSyst->GetRenderTarget(), m_renderTexture.Get());
		if (m_renderBlocks)
		{
			commandList->SetPipelineState(m_blocksRenderPipelineState.Get());
			for (int i = 0; i < blocksOrder.size(); i++)
			{
				commandList->DrawIndexedInstanced(36, 1, 0, 8 * blocksOrder[i].blockIndex, 0);
			}
		}

		m_d3dSyst->Execute();
		m_d3dSyst->PresentSimple();
	}
	
}

ComPtr<ID3D12Resource> VoxelPipeline::RegisterBlocksInfo(vector<BlockInfo>& blocksInfo)
{
	ComPtr<ID3D12Resource> blocksInfoRes = m_d3dSyst->CreateRWStructuredBuffer(&blocksInfo[0], sizeof(BlockInfo)*blocksInfo.size(), L"Blocks info");
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.Format = DXGI_FORMAT_UNKNOWN;
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavDesc.Buffer.FirstElement = 0;
	uavDesc.Buffer.NumElements = blocksInfo.size();
	uavDesc.Buffer.StructureByteStride = sizeof(BlockInfo);
	uavDesc.Buffer.CounterOffsetInBytes = 0;
	uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

	CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(m_blocksComputeSrvUavHeap->GetCPUDescriptorHandleForHeapStart(), COMPUTE_DESCRIPTORS::BLOCKS_INFO_UAV, m_srvUavDescriptorSize);
	m_d3dSyst->GetDevice()->CreateUnorderedAccessView(blocksInfoRes.Get(), nullptr, &uavDesc, uavHandle);

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.FirstElement = 0;
	srvDesc.Buffer.NumElements = blocksInfo.size();
	srvDesc.Buffer.StructureByteStride = sizeof(BlockInfo);
	srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

	CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_blocksComputeSrvUavHeap->GetCPUDescriptorHandleForHeapStart(), COMPUTE_DESCRIPTORS::BLOCKS_INFO_SRV, m_srvUavDescriptorSize);
	m_d3dSyst->GetDevice()->CreateShaderResourceView(blocksInfoRes.Get(), &srvDesc, srvHandle);

	return blocksInfoRes;
}

ComPtr<ID3D12Resource> VoxelPipeline::RegisterVoxels(vector<Voxel>& voxels)
{
	ComPtr<ID3D12Resource> voxelsRes = m_d3dSyst->CreateStructuredBuffer(&voxels[0], sizeof(Voxel)*voxels.size(), L"Voxels");
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.FirstElement = 0;
	srvDesc.Buffer.NumElements = voxels.size();
	srvDesc.Buffer.StructureByteStride = sizeof(Voxel);
	srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

	CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_blocksComputeSrvUavHeap->GetCPUDescriptorHandleForHeapStart(), COMPUTE_DESCRIPTORS::VOXELS_SRV, m_srvUavDescriptorSize);
	m_d3dSyst->GetDevice()->CreateShaderResourceView(voxelsRes.Get(), &srvDesc, srvHandle);
	return voxelsRes;
}

ComPtr<ID3D12Resource> VoxelPipeline::RegisterPalette(vector<uchar4>& palette)
{
	ComPtr<ID3D12Resource> paletteRes = m_d3dSyst->CreateTexture1D(palette, DXGI_FORMAT_R8G8B8A8_UNORM, L"Palette");
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
	srvDesc.Texture1D.MipLevels = 1;

	CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_srvUavHeapRender->GetCPUDescriptorHandleForHeapStart(), GRAPHICS_DESCRIPTORS::PALETTE_SRV, m_srvUavDescriptorSize);
	m_d3dSyst->GetDevice()->CreateShaderResourceView(paletteRes.Get(), &srvDesc, srvHandle);
	return paletteRes;
}

ComPtr<ID3D12Resource> VoxelPipeline::RegisterSegmentsOpacity(vector<float>& segmentsOpacity)
{
	ComPtr<ID3D12Resource> segmentsOpacityRes = m_d3dSyst->CreateTexture1D(segmentsOpacity, DXGI_FORMAT_R32_FLOAT, L"Sements opacity");
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
	srvDesc.Texture1D.MipLevels = 1;

	CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_srvUavHeapRender->GetCPUDescriptorHandleForHeapStart(), GRAPHICS_DESCRIPTORS::SEGMENTS_OPACITY_SRV, m_srvUavDescriptorSize);
	m_d3dSyst->GetDevice()->CreateShaderResourceView(segmentsOpacityRes.Get(), &srvDesc, srvHandle);
	return segmentsOpacityRes;
}

void VoxelPipeline::SetSegmentsOpacity(vector<float>& segmentsOpacity, ComPtr<ID3D12Resource>& segmentsOpacityRes)
{
	m_d3dSyst->CopyDataToGPU(segmentsOpacityRes, &segmentsOpacity[0], sizeof(float)*segmentsOpacity.size(),m_opacityUploadBuffer);
}


void VoxelPipeline::ComputeDetectBlocks(int voxelsCount, int3 dim, int blockSize, int3 dimBlocks, int3 min, int3 max, vector<BlockInfo>& blocksInfo, ComPtr<ID3D12Resource> blocksInfoRes)
{
	m_d3dSyst->Reset();
	int frameIndex = m_d3dSyst->GetFrameIndex();
	ComPtr<ID3D12GraphicsCommandList> commandList = m_d3dSyst->GetCommandList();
	commandList->SetPipelineState(m_blocksDetectionPipelineState.Get());
	commandList->SetComputeRootSignature(m_blocksComputeRootSignature.Get());
	ID3D12DescriptorHeap* heaps[] = { m_blocksComputeSrvUavHeap.Get() };
	commandList->SetDescriptorHeaps(_countof(heaps), heaps);

	ComputeBlocksCB computeConstantBuffer;
	computeConstantBuffer.min = { min.x, min.y, min.z, 0 };
	computeConstantBuffer.max = { max.x, max.y, max.z, 0 };
	computeConstantBuffer.dim = { dim.x, dim.y, dim.z, 0 };
	computeConstantBuffer.dimBlocks = { dimBlocks.x, dimBlocks.y, dimBlocks.z, 0 };
	computeConstantBuffer.voxelsCount = voxelsCount;
	computeConstantBuffer.blockSize = blockSize;
	int computeBlocksCount = ceil(sqrt(voxelsCount));
	computeBlocksCount = ceil(computeBlocksCount / 32.0);
	computeConstantBuffer.computeBlocksCount = computeBlocksCount;

	memcpy(m_cbvGPUAddress[frameIndex], &computeConstantBuffer, sizeof(computeConstantBuffer));
	commandList->SetComputeRootConstantBufferView(0, m_constantBufferUploadHeaps[frameIndex]->GetGPUVirtualAddress());
	commandList->SetComputeRootDescriptorTable(1, m_blocksComputeSrvUavHeap->GetGPUDescriptorHandleForHeapStart());
	commandList->Dispatch(computeBlocksCount, computeBlocksCount, 1);
	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(blocksInfoRes.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));

	m_d3dSyst->Execute();
	m_d3dSyst->Wait();
	m_d3dSyst->CopyDataFromGPU(blocksInfoRes, &blocksInfo[0], sizeof(BlockInfo)*blocksInfo.size());
}

void VoxelPipeline::RegisterBlocks(int overlap, int3 dimBlocks, int blockSize, vector<BlockInfo>& blocksInfo, ComPtr<ID3D12Resource>& blocksRes, vector<ComPtr<ID3D12Resource>>& texturesRes, ComPtr<ID3D12Resource>& blocksIndexesRes, vector<BlockPositionInfo>& blocksPosInfo, vector<BlockPriorityInfo>& blocksPriorInfo)
{
	texturesRes.clear();
	blocksPosInfo.clear();
	blocksPriorInfo.clear();
	vector<Block> blocks;
	vector<int> blocksIndexes;
	DXGI_FORMAT format = DXGI_FORMAT_R8G8_UINT;
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.Format = format;
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = format;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
	srvDesc.Texture3D.MipLevels = 1;
	for (int i = 0; i < blocksInfo.size(); i++)
	{
		int3 block3dIndex;
		block3dIndex.z = i / (dimBlocks.x * dimBlocks.y);
		int tmp = i % (dimBlocks.x * dimBlocks.y);
		block3dIndex.y = tmp / dimBlocks.x;
		block3dIndex.x = tmp % dimBlocks.x;
		BlockPositionInfo blockPositionInfo;
		blockPositionInfo.block3dIndex = block3dIndex;
		blockPositionInfo.distance = 0;
		blockPositionInfo.position = Vector3(block3dIndex.x + 0.5f, block3dIndex.y + 0.5f, block3dIndex.z + 0.5f) * blockSize;
		blocksPosInfo.push_back(blockPositionInfo);
		if ((blocksInfo[i].max.x >= blocksInfo[i].min.x) && (blocksInfo[i].max.y >= blocksInfo[i].min.y) && (blocksInfo[i].max.z >= blocksInfo[i].min.z))
		{
			blocksIndexes.push_back(blocks.size());
			blocks.emplace_back(blocksInfo[i].min, blocksInfo[i].max, overlap);
			int3 dim = { 1 + (blocksInfo[i].max.x - blocksInfo[i].min.x + 2 * overlap), 1 + (blocksInfo[i].max.y - blocksInfo[i].min.y + 2 * overlap), 1 + (blocksInfo[i].max.z - blocksInfo[i].min.z + 2 * overlap) };
			uavDesc.Texture3D.WSize = dim.z;
			ComPtr<ID3D12Resource> textureRes = m_d3dSyst->CreateRWTexture3D(dim, format, L"3D texture");
			CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(m_blocksComputeSrvUavHeap->GetCPUDescriptorHandleForHeapStart(), COMPUTE_DESCRIPTORS::TEXTURES_3D_UAV_ARRAY + texturesRes.size(), m_srvUavDescriptorSize);
			m_d3dSyst->GetDevice()->CreateUnorderedAccessView(textureRes.Get(), nullptr, &uavDesc, uavHandle);
			CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_srvUavHeapRender->GetCPUDescriptorHandleForHeapStart(), GRAPHICS_DESCRIPTORS::TEXTURES_3D_SRV_ARRAY + texturesRes.size(), m_srvUavDescriptorSize);
			m_d3dSyst->GetDevice()->CreateShaderResourceView(textureRes.Get(), &srvDesc, srvHandle);
			texturesRes.push_back(textureRes);
			BlockPriorityInfo blockPriorInfo;
			blockPriorInfo.block3dIndex = block3dIndex;
			blockPriorInfo.blockIndex = blocksPriorInfo.size();
			blockPriorInfo.priority = 0;
			blocksPriorInfo.push_back(blockPriorInfo);
		}
		else
		{
			blocksIndexes.push_back(-1);
		}
	}
	blocksRes = m_d3dSyst->CreateVertexBuffer(&blocks[0], sizeof(Block)*blocks.size(), L"Blocks vertex buffer");
	blocksIndexesRes = m_d3dSyst->CreateStructuredBuffer(&blocksIndexes[0], sizeof(int)*blocksIndexes.size(), L"Blocks indexes");
	D3D12_SHADER_RESOURCE_VIEW_DESC srvIndexesDesc = {};
	srvIndexesDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvIndexesDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvIndexesDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvIndexesDesc.Buffer.FirstElement = 0;
	srvIndexesDesc.Buffer.NumElements = blocksIndexes.size();
	srvIndexesDesc.Buffer.StructureByteStride = sizeof(int);
	srvIndexesDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_blocksComputeSrvUavHeap->GetCPUDescriptorHandleForHeapStart(), COMPUTE_DESCRIPTORS::BLOCKS_INDEXES_SRV, m_srvUavDescriptorSize);
	m_d3dSyst->GetDevice()->CreateShaderResourceView(blocksIndexesRes.Get(), &srvIndexesDesc, srvHandle);
}

void VoxelPipeline::ComputeFillBlocks(int voxelsCount, int texturesCount, int3 dim, int blockSize, int3 dimBlocks, int3 min, int3 max, int overlap, vector<ComPtr<ID3D12Resource>>& texturesRes)
{
	m_d3dSyst->Reset();
	int frameIndex = m_d3dSyst->GetFrameIndex();
	ComPtr<ID3D12GraphicsCommandList> commandList = m_d3dSyst->GetCommandList();
	commandList->SetPipelineState(m_blocksFillingPipelineState.Get());
	commandList->SetComputeRootSignature(m_blocksComputeRootSignature.Get());
	ID3D12DescriptorHeap* heaps[] = { m_blocksComputeSrvUavHeap.Get() };
	commandList->SetDescriptorHeaps(_countof(heaps), heaps);
	int computeBlocksCount = ceil(sqrt(voxelsCount));
	computeBlocksCount = ceil(computeBlocksCount / 32.0);
	ComputeBlocksCB computeConstantBuffer;
	computeConstantBuffer.min = { min.x, min.y, min.z, 0 };
	computeConstantBuffer.max = { max.x, max.y, max.z, 0 };
	computeConstantBuffer.dim = { dim.x, dim.y, dim.z, 0 };
	computeConstantBuffer.dimBlocks = { dimBlocks.x, dimBlocks.y, dimBlocks.z, 0 };
	computeConstantBuffer.voxelsCount = voxelsCount;
	computeConstantBuffer.blockSize = blockSize;
	computeConstantBuffer.computeBlocksCount = computeBlocksCount;
	computeConstantBuffer.overlap = overlap;
	memcpy(m_cbvGPUAddress[frameIndex], &computeConstantBuffer, sizeof(computeConstantBuffer));
	commandList->SetComputeRootConstantBufferView(0, m_constantBufferUploadHeaps[frameIndex]->GetGPUVirtualAddress());
	commandList->SetComputeRootDescriptorTable(1, m_blocksComputeSrvUavHeap->GetGPUDescriptorHandleForHeapStart());
	commandList->Dispatch(computeBlocksCount, computeBlocksCount, 1);
	vector<CD3DX12_RESOURCE_BARRIER> transitions;
	for (int i = 0; i < texturesCount; i++)
	{
		transitions.push_back(CD3DX12_RESOURCE_BARRIER::Transition(texturesRes[i].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
	}
	commandList->ResourceBarrier(texturesCount, &transitions[0]);
	m_d3dSyst->Execute();
	m_d3dSyst->Wait();
}

void VoxelPipeline::SetStepSize(float voxelSize, float ratio)
{
	m_renderingCB.stepSize = voxelSize * ratio;
	m_renderingCB.stepRatio = ratio;
}

void VoxelPipeline::SetInterpolationMode(INTERPOLATION_MODE mode)
{
	switch (mode)
	{
	case VoxelPipeline::INTERPOLATION_MODE_NONE:
		m_selectedRCPipelineState = m_rayCastingPipelineState;
		break;
	case VoxelPipeline::INTERPOLATION_MODE_TRILINEAR:
		m_selectedRCPipelineState = m_rayCastingTrilinearPipelineState;
		break;
	default:
		break;
	}
}

void VoxelPipeline::SetBlocksVisiblity(bool isVisible)
{
	m_renderBlocks = isVisible;
}

