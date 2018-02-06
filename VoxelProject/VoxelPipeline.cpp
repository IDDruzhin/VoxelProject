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
		CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
		//ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 1, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);
		ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, -1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);

		CD3DX12_ROOT_PARAMETER1 rootParameters[2];
		rootParameters[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_VERTEX);
		//rootParameters[1].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);
		rootParameters[1].InitAsDescriptorTable(2, &ranges[0], D3D12_SHADER_VISIBILITY_PIXEL);

		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
		rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		ID3DBlob *signature;
		ID3DBlob *error;
		ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, m_d3dSyst->GetFeatureData().HighestVersion, &signature, &error));
		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_meshRootSignature)));

		ComPtr<ID3DBlob> vertexShader;
		ComPtr<ID3DBlob> pixelShader;
#if defined(_DEBUG)
		UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION | D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES;
#else
		UINT compileFlags = 0;
#endif
		//UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
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

	/// Blocks detection and filling compute pipeline
	{
		D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
		srvUavHeapDesc.NumDescriptors = 10000;
		srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		d3dSyst->GetDevice()->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&m_blocksComputeSrvUavHeap));

		CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE); //t0-t2
		ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, -1, 1, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE); //u1-...

		CD3DX12_ROOT_PARAMETER1 rootParameters[2];
		rootParameters[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_ALL);
		//rootParameters[1].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);
		rootParameters[1].InitAsDescriptorTable(2, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);

		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
		rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

		ID3DBlob *signature;
		ID3DBlob *error;
		ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, m_d3dSyst->GetFeatureData().HighestVersion, &signature, &error));
		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_blocksComputeRootSignature)));

		ComPtr<ID3DBlob> computeShader;
#if defined(_DEBUG)
		UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION | D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES;
#else
		UINT compileFlags = 0;
#endif
		//UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
		ThrowIfFailed(D3DCompileFromFile(L"BlocksDetectionCS.hlsl", nullptr, nullptr, "main", "cs_5_1", compileFlags, 0, &computeShader, nullptr));
		D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
		computePsoDesc.pRootSignature = m_blocksComputeRootSignature.Get();
		computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());
		ThrowIfFailed(d3dSyst->GetDevice()->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_blocksDetectionPipelineState)));

		/*
		ID3DBlob* err;
		char* chErr;
		D3DCompileFromFile(L"BlocksFillingCS.hlsl", nullptr, nullptr, "main", "cs_5_1", compileFlags, 0, &computeShader, &err);
		chErr = (char*)err->GetBufferPointer();
		*/
		ThrowIfFailed(D3DCompileFromFile(L"BlocksFillingCS.hlsl", nullptr, nullptr, "main", "cs_5_1", compileFlags, 0, &computeShader, nullptr));
		//D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
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
			&CD3DX12_RESOURCE_DESC::Buffer(1024 * 64), //Size. Must be a multiple of 64KB for constant buffers
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_constantBufferUploadHeaps[i])));
		//m_constantBufferUploadHeaps[i]->SetName(L"Constant Buffer Upload Resource Heap");
		CD3DX12_RANGE readRange(0, 0);    //No reading on CPU
		ThrowIfFailed(m_constantBufferUploadHeaps[i]->Map(0, &readRange, reinterpret_cast<void**>(&m_cbvGPUAddress[i])));
	}

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
		CD3DX12_RESOURCE_DESC textureDesc = CD3DX12_RESOURCE_DESC::Tex2D(swapChainDesc.BufferDesc.Format, swapChainDesc.BufferDesc.Width, swapChainDesc.BufferDesc.Height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

		ThrowIfFailed(m_d3dSyst->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), //A default heap
			D3D12_HEAP_FLAG_NONE, //No flags
			&textureDesc,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			nullptr,
			IID_PPV_ARGS(&m_renderTexture)));
		//m_renderTexture->SetName(L"Render texture");

		D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
		uavDesc.Format = textureDesc.Format;
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		//uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

		CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(m_srvUavHeapRender->GetCPUDescriptorHandleForHeapStart());
		m_d3dSyst->GetDevice()->CreateUnorderedAccessView(m_renderTexture.Get(), nullptr, &uavDesc, uavHandle);

		CD3DX12_CPU_DESCRIPTOR_HANDLE cpuWriteUavHandle(m_rtvHeapRender->GetCPUDescriptorHandleForHeapStart());
		m_d3dSyst->GetDevice()->CreateUnorderedAccessView(m_renderTexture.Get(), nullptr, &uavDesc, cpuWriteUavHandle);
	}
	m_srvUavDescriptorSize = m_d3dSyst->GetDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

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
		ID3D12DescriptorHeap* heaps[] = { m_srvUavHeapRender.Get() };
		commandList->SetDescriptorHeaps(_countof(heaps), heaps);

		commandList->SetPipelineState(m_meshPipelineState.Get());
		commandList->SetGraphicsRootSignature(m_meshRootSignature.Get());
		commandList->RSSetViewports(1, &m_viewport);
		commandList->RSSetScissorRects(1, &m_scissorRect);
		commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		memcpy(m_cbvGPUAddress[frameIndex], &m_renderingCB, sizeof(m_renderingCB));
		commandList->SetGraphicsRootConstantBufferView(0, m_constantBufferUploadHeaps[frameIndex]->GetGPUVirtualAddress());
		//CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(m_srvUavHeapRender->GetGPUDescriptorHandleForHeapStart());
		commandList->SetGraphicsRootDescriptorTable(1, m_srvUavHeapRender->GetGPUDescriptorHandleForHeapStart());
		
		const FLOAT clearVal[4] = { 0.0f,0.0f,0.0f,0.0f };
		CD3DX12_GPU_DESCRIPTOR_HANDLE gpuH(m_srvUavHeapRender->GetGPUDescriptorHandleForHeapStart());
		//auto CpuH = m_d3dSyst->GetRtvCPUHandle();
		CD3DX12_CPU_DESCRIPTOR_HANDLE cpuH(m_rtvHeapRender->GetCPUDescriptorHandleForHeapStart());
		commandList->ClearUnorderedAccessViewFloat(gpuH, cpuH, m_renderTexture.Get(), &clearVal[0],0,nullptr);
		//commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_renderTexture.Get()));
		
		commandList->IASetVertexBuffers(0, 1, &(voxObj->GetBlocksVBV()));
		//commandList->DrawInstanced(3, 1, 0, 0);
		commandList->IASetIndexBuffer(&m_blockIndexBufferView);
		//commandList->DrawIndexedInstanced(36, 1, 0, 0, 0);
		//commandList->DrawIndexedInstanced(36, 1, 0, 0, 36*100);
		//commandList->DrawIndexedInstanced(36, voxObj->GetBlocksCount(), 0, 0, 8);
		//commandList->DrawIndexedInstanced(36, voxObj->GetBlocksCount(), 0, 0, 8);
		//commandList->DrawIndexedInstanced(36, 1, 0, 8 * 100, 0);
		vector<BlockPositionInfo> blocksOrder = voxObj->CalculatePriorities(Vector3(0, 0, 0));
		for (int i = 0; i < blocksOrder.size(); i++)
		{
			commandList->DrawIndexedInstanced(36, 1, 0, 8 * blocksOrder[i].blockIndex, 0);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_renderTexture.Get()));
		}
		/*
		for (int i = 0; i < voxObj->GetBlocksCount(); i++)
		{
			//commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_renderTexture.Get()));
			commandList->DrawIndexedInstanced(36, 1, 0, 8*i, 0);
			//ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_renderTexture.Get()));
		}
		*/
		commandList->CopyResource(m_d3dSyst->GetRenderTarget(), m_renderTexture.Get());
		m_d3dSyst->ExecuteGraphics();
		//m_d3dSyst->Reset();
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

	CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandleComputeBorder(m_blocksComputeSrvUavHeap->GetCPUDescriptorHandleForHeapStart(), 1, m_srvUavDescriptorSize);
	m_d3dSyst->GetDevice()->CreateUnorderedAccessView(blocksInfoRes.Get(), nullptr, &uavDesc, uavHandleComputeBorder);
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

	CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandleGraphic(m_blocksComputeSrvUavHeap->GetCPUDescriptorHandleForHeapStart(), 0, m_srvUavDescriptorSize);
	m_d3dSyst->GetDevice()->CreateShaderResourceView(voxelsRes.Get(), &srvDesc, srvHandleGraphic);
	return voxelsRes;
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
	//ID3D12Resource *pUavResource = voxelObjects[SelectedObject].GetBorderMaskRes();
	//ID3D12Resource *pUavResource = voxelObjects[SelectedObject].GetActivityMaskRes();
	//commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pUavResource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

	//CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(m_srvHeapComputeActivity->GetGPUDescriptorHandleForHeapStart(), 0, m_srvUavDescriptorSize);
	//CD3DX12_GPU_DESCRIPTOR_HANDLE uavHandle(m_srvHeapComputeActivity->GetGPUDescriptorHandleForHeapStart(), 2, m_srvUavDescriptorSize);
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
	//int MaxObjectCount = 1;
	memcpy(m_cbvGPUAddress[frameIndex], &computeConstantBuffer, sizeof(computeConstantBuffer));
	commandList->SetComputeRootConstantBufferView(0, m_constantBufferUploadHeaps[frameIndex]->GetGPUVirtualAddress());
	commandList->SetComputeRootDescriptorTable(1, m_blocksComputeSrvUavHeap->GetGPUDescriptorHandleForHeapStart());
	//commandList->SetComputeRootDescriptorTable(1, uavHandle);
	//commandList->SetComputeRootDescriptorTable(1)
	//commandList->SetComputeRootShaderResourceView(0, voxelObjects[SelectedObject].GetVoxelDataGPUAddress());
	//commandList->SetComputeRootShaderResourceView(1, voxelObjects[SelectedObject].GetSegmentMaskBitSetSize());
	//commandList->SetComputeRootUnorderedAccessView(2, pUavResource->GetGPUVirtualAddress());
	//commandList->SetComputeRootUnorderedAccessView(2, voxelObjects[SelectedObject].GetBorderMaskGPUAddress());
	//commandList->SetComputeRootUnorderedAccessView(2, voxelObjects[SelectedObject].GetActivityMaskGPUAddress());
	//int Count = static_cast<int>(ceil(voxelObjects[SelectedObject].GetTotalVoxelsCount() / (4.0f * 256.0f)));
	//commandList->Dispatch(Count, 1, 1);

	//commandList->Dispatch(static_cast<int>(ceil(93420279 / (2*256.0f))), 1, 1);
	//commandList->Dispatch(10, 1, 1);
	//commandList->Dispatch(static_cast<int>(ceil(1000 / 256.0f)), 1, 1);
	//commandList->Dispatch(static_cast<int>(ceil(voxelObjects[SelectedObject].GetTotalVoxelsCount() / 256.0f)), 1, 1);
	commandList->Dispatch(computeBlocksCount, computeBlocksCount, 1);
	//commandList->Dispatch(1, 1, 1);
	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(blocksInfoRes.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
	//m_d3dSyst->UpdatePipelineAndClear(Vector3(0, 0, 0));
	//m_d3dSyst->ExecuteGraphics();
	//m_d3dSyst->PresentSimple();
	m_d3dSyst->Execute();
	m_d3dSyst->Wait();
	m_d3dSyst->CopyDataFromGPU(blocksInfoRes, &blocksInfo[0], sizeof(BlockInfo)*blocksInfo.size());
}

void VoxelPipeline::RegisterBlocks(int overlap, int3 dimBlocks, vector<BlockInfo>& blocksInfo, ComPtr<ID3D12Resource>& blocksRes, vector<ComPtr<ID3D12Resource>>& texturesRes, ComPtr<ID3D12Resource>& blocksIndexesRes, vector<BlockPositionInfo>& blocksPosInfo)
{
	texturesRes.clear();
	blocksPosInfo.clear();
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
	//srvDesc.Buffer.FirstElement = 0;
	//srvDesc.Buffer.NumElements = ElementsCount;
	//srvDesc.Buffer.StructureByteStride = sizeof(T);
	//srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

	CD3DX12_CPU_DESCRIPTOR_HANDLE cpuWriteUavHandle(m_rtvHeapRender->GetCPUDescriptorHandleForHeapStart());
	m_d3dSyst->GetDevice()->CreateUnorderedAccessView(m_renderTexture.Get(), nullptr, &uavDesc, cpuWriteUavHandle);
	for (int i = 0; i < blocksInfo.size(); i++)
	{
		if ((blocksInfo[i].max.x >= blocksInfo[i].min.x) && (blocksInfo[i].max.y >= blocksInfo[i].min.y) && (blocksInfo[i].max.z >= blocksInfo[i].min.z))
		{
			blocksIndexes.push_back(blocks.size());
			blocks.emplace_back(blocksInfo[i].min, blocksInfo[i].max, overlap);
			int3 dim = { (blocksInfo[i].max.x - blocksInfo[i].min.x + 2 * overlap), (blocksInfo[i].max.y - blocksInfo[i].min.y + 2 * overlap), (blocksInfo[i].max.z - blocksInfo[i].min.z + 2 * overlap) };
			ComPtr<ID3D12Resource> textureRes = m_d3dSyst->CreateRWTexture3D(dim, format, L"3D texture");

			CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(m_blocksComputeSrvUavHeap->GetCPUDescriptorHandleForHeapStart(), COMPUTE_DESCRIPTORS::TEXTURES_3D_UAV_ARRAY + texturesRes.size(), m_srvUavDescriptorSize);
			m_d3dSyst->GetDevice()->CreateUnorderedAccessView(textureRes.Get(), nullptr, &uavDesc, uavHandle);

			CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_blocksComputeSrvUavHeap->GetCPUDescriptorHandleForHeapStart(), GRAPHICS_DESCRIPTORS::TEXTURES_3D_SRV_ARRAY + texturesRes.size(), m_srvUavDescriptorSize);
			m_d3dSyst->GetDevice()->CreateShaderResourceView(textureRes.Get(), &srvDesc, srvHandle);

			texturesRes.push_back(textureRes);

			int3 block3dIndex;
			block3dIndex.z = i / (dimBlocks.x * dimBlocks.y);
			int tmp = i % (dimBlocks.x * dimBlocks.y);
			block3dIndex.y = tmp / dimBlocks.x;
			block3dIndex.x = tmp % dimBlocks.x;
			//blocks3dIndexes.push_back(block3dIndex);
			//blocksPositions.emplace_back(block3dIndex.x + 0.5f, block3dIndex.y + 0.5f, block3dIndex.z + 0.5f);
			BlockPositionInfo blockPositionInfo;
			blockPositionInfo.block3dIndex = block3dIndex;
			blockPositionInfo.blockIndex = blocksPosInfo.size();
			blockPositionInfo.distance = 0;
			blockPositionInfo.position = Vector3(block3dIndex.x + 0.5f, block3dIndex.y + 0.5f, block3dIndex.z + 0.5f);
			blockPositionInfo.priority = 0;
			blocksPosInfo.push_back(blockPositionInfo);
		}
		else
		{
			blocksIndexes.push_back(-1);
		}
	}
	blocksIndexesRes = m_d3dSyst->CreateVertexBuffer(&blocksIndexes[0], sizeof(int)*blocksIndexes.size(), L"Blocks indexes");
}

void VoxelPipeline::ComputeFillBlocks(int voxelsCount, int3 dim, int blockSize, int3 dimBlocks, int3 min, int3 max, int overlap, vector<ComPtr<ID3D12Resource>>& texturesRes)
{
	m_d3dSyst->Reset();
	int frameIndex = m_d3dSyst->GetFrameIndex();
	ComPtr<ID3D12GraphicsCommandList> commandList = m_d3dSyst->GetCommandList();
	commandList->SetPipelineState(m_blocksDetectionPipelineState.Get());
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
	for (int i = 0; i < texturesRes.size(); i++)
	{
		transitions.push_back(CD3DX12_RESOURCE_BARRIER::Transition(texturesRes[i].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
	}
	commandList->ResourceBarrier(transitions.size(), &transitions[0]);
	m_d3dSyst->Execute();
	m_d3dSyst->Wait();
}

