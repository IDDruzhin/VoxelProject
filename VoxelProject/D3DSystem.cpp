#include "stdafx.h"
#include "D3DSystem.h"


D3DSystem::D3DSystem(HWND hWnd, int width, int height)
{
	///Device
	UINT dxgiFactoryFlags = 0;
	/*
	ID3D12Debug *debugController;
	ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
	{
		debugController->EnableDebugLayer();
		dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
	}
	*/
#if defined(_DEBUG)
	{
		ComPtr<ID3D12Debug> debugController;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
		{
			debugController->EnableDebugLayer();
			dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
		}
	}
#endif

	ComPtr<IDXGIFactory4> factory;
	ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

	IDXGIAdapter1* adapter;
	int adapterIndex = 0;

	while (factory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND)
	{
		DXGI_ADAPTER_DESC1 desc;
		adapter->GetDesc1(&desc);

		if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
		{
			adapterIndex++;
			continue;
		}

		if (SUCCEEDED(D3D12CreateDevice(adapter,D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
		{
			ThrowIfFailed(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));
			break;
		}

		adapterIndex++;
	}

	m_featureData = {};
	m_featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

	if (FAILED(m_device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &m_featureData, sizeof(m_featureData))))
	{
		m_featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
	}

	///Queque

	D3D12_COMMAND_QUEUE_DESC cqDesc = {};
	ThrowIfFailed(m_device->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&m_commandQueue)));

	///Swap chain

	DXGI_MODE_DESC backBufferDesc = {};
	backBufferDesc.Width = width;
	backBufferDesc.Height = height;
	backBufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	DXGI_SAMPLE_DESC sampleDesc = {};
	sampleDesc.Count = 1; // multisample count

	DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
	swapChainDesc.BufferCount = FRAMEBUFFERCOUNT;
	swapChainDesc.BufferDesc = backBufferDesc;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.OutputWindow = hWnd;
	swapChainDesc.SampleDesc = sampleDesc;
	swapChainDesc.Windowed = true;
	//swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

	IDXGISwapChain* tempSwapChain;
	ThrowIfFailed(factory->CreateSwapChain(
		m_commandQueue.Get(),
		&swapChainDesc,
		&tempSwapChain
	));

	m_swapChain = static_cast<IDXGISwapChain3*>(tempSwapChain);

	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
	//m_swapChainEvent = m_swapChain->GetFrameLatencyWaitableObject();

	///Back buffers descriptor heap

	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
	rtvHeapDesc.NumDescriptors = FRAMEBUFFERCOUNT;
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	ThrowIfFailed(m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvDescriptorHeap)));
	m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

	for (int i = 0; i < FRAMEBUFFERCOUNT; i++)
	{
		ThrowIfFailed(m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_renderTargets[i])));
		m_device->CreateRenderTargetView(m_renderTargets[i].Get(), nullptr, rtvHandle);
		rtvHandle.Offset(1, m_rtvDescriptorSize);
	}

	D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
	dsvHeapDesc.NumDescriptors = 1;
	dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	ThrowIfFailed(m_device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_dsDescriptorHeap)));
	D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
	depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

	D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
	depthOptimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
	depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
	depthOptimizedClearValue.DepthStencil.Stencil = 0;

	ThrowIfFailed(m_device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
		D3D12_RESOURCE_STATE_DEPTH_WRITE,
		&depthOptimizedClearValue,
		IID_PPV_ARGS(&m_depthStencilBuffer)
	));
	//m_dsDescriptorHeap->SetName(L"Depth/Stencil Resource Heap");
	m_device->CreateDepthStencilView(m_depthStencilBuffer.Get(), &depthStencilDesc, m_dsDescriptorHeap->GetCPUDescriptorHandleForHeapStart());


	///Command allocators

	for (int i = 0; i < FRAMEBUFFERCOUNT; i++)
	{
		ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator[i])));
	}

	ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator[0].Get(), NULL, IID_PPV_ARGS(&m_commandList)));
	m_commandList->Close();

	///Fences
	for (int i = 0; i < FRAMEBUFFERCOUNT; i++)
	{
		ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence[i])));
		m_fenceValue[i] = 0;
	}
	m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	if (m_fenceEvent == nullptr)
	{
		ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
	}
}


D3DSystem::~D3DSystem()
{
	OnDestroy();
}

ComPtr<ID3D12Device> D3DSystem::GetDevice()
{
	return m_device;
}

DXGI_SWAP_CHAIN_DESC D3DSystem::GetSwapChainDesc()
{
	DXGI_SWAP_CHAIN_DESC desc;
	m_swapChain->GetDesc(&desc);
	return desc;
}

ComPtr<ID3D12GraphicsCommandList> D3DSystem::GetCommandList()
{
	return m_commandList;
}

void D3DSystem::Reset()
{
	//WaitForSingleObjectEx(m_swapChainEvent, 100, FALSE);
	//WaitForPreviousFrame();
	Wait();
	ThrowIfFailed(m_commandAllocator[m_frameIndex]->Reset());
	ThrowIfFailed(m_commandList->Reset(m_commandAllocator[m_frameIndex].Get(), NULL));
	//PIXBeginEvent(m_commandQueue, 0, L"Render");
}

void D3DSystem::Execute()
{
	m_commandList->Close();
	ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

	m_fenceValue[m_frameIndex]++;
	ThrowIfFailed(m_commandQueue->Signal(m_fence[m_frameIndex].Get(), m_fenceValue[m_frameIndex]));
}

void D3DSystem::ExecuteGraphics()
{
	//CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex, m_rtvDescriptorSize);
	//float clearColor[4] = { 0, 1, 0, 1.0f };
	//m_commandList->ClearRenderTargetView(rtvHandle, (float*)&C, 0, nullptr);
	//m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

	//m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));
	m_commandList->Close();
	ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

	m_fenceValue[m_frameIndex]++;
	ThrowIfFailed(m_commandQueue->Signal(m_fence[m_frameIndex].Get(), m_fenceValue[m_frameIndex]));
}

void D3DSystem::UpdatePipelineAndClear(Vector3 Bg)
{
	//m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex, m_rtvDescriptorSize);
	//CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(m_dsDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	//m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
	m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
	//float clearColor[] = { Bg.x, Bg.y, Bg.z, 1.0f };
	//m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	//m_commandList->ClearDepthStencilView(m_dsDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
}

/*
void D3DSystem::Wait()
{
	WaitForPreviousFrame();
	HRESULT hr = m_commandQueue->Signal(m_fence[m_frameIndex].Get(), m_fenceValue[m_frameIndex]);
}
*/

/*
void D3DSystem::Wait()
{
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
	m_fenceValue[m_frameIndex]++;
	ThrowIfFailed(m_commandQueue->Signal(m_fence[m_frameIndex].Get(), m_fenceValue[m_frameIndex]));
	ThrowIfFailed(m_fence[m_frameIndex]->SetEventOnCompletion(m_fenceValue[m_frameIndex], m_fenceEvent));
	WaitForSingleObject(m_fenceEvent, INFINITE);
}
*/

void D3DSystem::PresentSimple()
{
	// present the current backbuffer
	ThrowIfFailed(m_swapChain->Present(0, 0));
	//ThrowIfFailed(m_swapChain->Present(1, 0));
	//m_swapChain->Present(4, 0);
	//ThrowIfFailed(m_swapChain->Present1(1, 0, nullptr));
}

ID3D12Resource * D3DSystem::GetRenderTarget()
{
	return m_renderTargets[m_frameIndex].Get();
}

CD3DX12_CPU_DESCRIPTOR_HANDLE D3DSystem::GetRtvCPUHandle()
{
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	return rtvHandle;
}


void D3DSystem::OnDestroy()
{
	///Wait for completion
	for (int i = 0; i < FRAMEBUFFERCOUNT; ++i)
	{
		m_frameIndex = i;
		if (m_fence[m_frameIndex]->GetCompletedValue() < m_fenceValue[m_frameIndex])
		{
			m_fence[m_frameIndex]->SetEventOnCompletion(m_fenceValue[m_frameIndex], m_fenceEvent);
			WaitForSingleObject(m_fenceEvent, INFINITE);
		}
	}

	BOOL fs = false;
	if (m_swapChain->GetFullscreenState(&fs, NULL))
		m_swapChain->SetFullscreenState(false, NULL);

	///ComPtr auto Release
}

int D3DSystem::GetFrameIndex()
{
	return m_swapChain->GetCurrentBackBufferIndex();
}

D3D12_FEATURE_DATA_ROOT_SIGNATURE D3DSystem::GetFeatureData()
{
	return m_featureData;
}

//void D3DSystem::WaitForPreviousFrame()
void D3DSystem::Wait()
{
	//m_fenceValue[m_frameIndex]++;
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
	if (m_fence[m_frameIndex]->GetCompletedValue() < m_fenceValue[m_frameIndex])
	{
		ThrowIfFailed(m_fence[m_frameIndex]->SetEventOnCompletion(m_fenceValue[m_frameIndex], m_fenceEvent));
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}
	//m_fenceValue[m_frameIndex]++;
}
