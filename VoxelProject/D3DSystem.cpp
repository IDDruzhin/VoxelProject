#include "stdafx.h"
#include "D3DSystem.h"


D3DSystem::D3DSystem()
{
	m_featureData = {};
}


D3DSystem::~D3DSystem()
{
}

bool D3DSystem::InitD3D(HWND hWnd, int Width, int Height)
{
	HRESULT hr;
	///Device
	UINT dxgiFactoryFlags = 0;
	ID3D12Debug *debugController;
	hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
	{
		debugController->EnableDebugLayer();
		dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
	}

	IDXGIFactory4* dxgiFactory;
	hr = CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&dxgiFactory));
	if (FAILED(hr))
	{
		return false;
	}

	IDXGIAdapter1* adapter;
	int adapterIndex = 0;
	bool adapterFound = false;

	while (dxgiFactory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND)
	{
		DXGI_ADAPTER_DESC1 desc;
		adapter->GetDesc1(&desc);

		if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
		{
			adapterIndex++;
			continue;
		}

		hr = D3D12CreateDevice(
			adapter,
			D3D_FEATURE_LEVEL_11_0,
			IID_PPV_ARGS(&m_device)
		);
		if (SUCCEEDED(hr))
		{
			adapterFound = true;
			break;
		}

		adapterIndex++;
	}

	if (!adapterFound)
	{
		return false;
	}

	m_featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

	if (FAILED(m_device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &m_featureData, sizeof(m_featureData))))
	{
		m_featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
	}

	///Queque

	D3D12_COMMAND_QUEUE_DESC cqDesc = {};
	hr = m_device->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&m_commandQueue));
	if (FAILED(hr))
	{
		return false;
	}

	///Swap chain

	DXGI_MODE_DESC backBufferDesc = {};
	backBufferDesc.Width = Width; 
	backBufferDesc.Height = Height;
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
	swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

	IDXGISwapChain* tempSwapChain;
	dxgiFactory->CreateSwapChain(
		m_commandQueue,
		&swapChainDesc,
		&tempSwapChain
	);

	m_swapChain = static_cast<IDXGISwapChain3*>(tempSwapChain);

	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
	m_swapChainEvent = m_swapChain->GetFrameLatencyWaitableObject();

	///Back buffers descriptor heap

	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
	rtvHeapDesc.NumDescriptors = FRAMEBUFFERCOUNT;
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	hr = m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvDescriptorHeap));
	if (FAILED(hr))
	{
		return false;
	}
	m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

	for (int i = 0; i < FRAMEBUFFERCOUNT; i++)
	{
		hr = m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_renderTargets[i]));
		if (FAILED(hr))
		{
			return false;
		}
		m_device->CreateRenderTargetView(m_renderTargets[i], nullptr, rtvHandle);
		rtvHandle.Offset(1, m_rtvDescriptorSize);
	}

	D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
	dsvHeapDesc.NumDescriptors = 1;
	dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	hr = m_device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_dsDescriptorHeap));
	if (FAILED(hr))
	{
		return false;
	}
	D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
	depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

	D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
	depthOptimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
	depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
	depthOptimizedClearValue.DepthStencil.Stencil = 0;

	m_device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, Width, Height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
		D3D12_RESOURCE_STATE_DEPTH_WRITE,
		&depthOptimizedClearValue,
		IID_PPV_ARGS(&m_depthStencilBuffer)
	);
	m_dsDescriptorHeap->SetName(L"Depth/Stencil Resource Heap");
	m_device->CreateDepthStencilView(m_depthStencilBuffer, &depthStencilDesc, m_dsDescriptorHeap->GetCPUDescriptorHandleForHeapStart());


	///Command allocators

	for (int i = 0; i < FRAMEBUFFERCOUNT; i++)
	{
		hr = m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator[i]));
		if (FAILED(hr))
		{
			return false;
		}
	}

	hr = m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator[0], NULL, IID_PPV_ARGS(&m_commandList));
	if (FAILED(hr))
	{
		return false;
	}
	m_commandList->Close();

	///Fences
	for (int i = 0; i < FRAMEBUFFERCOUNT; i++)
	{
		hr = m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence[i]));
		if (FAILED(hr))
		{
			return false;
		}
		m_fenceValue[i] = 0;
	}
	m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	if (m_fenceEvent == nullptr)
	{
		return false;
	}
	return true;
}

ID3D12Device * D3DSystem::GetDevice()
{
	return m_device;
}

DXGI_SWAP_CHAIN_DESC D3DSystem::GetSwapChainDesc()
{
	DXGI_SWAP_CHAIN_DESC desc;
	m_swapChain->GetDesc(&desc);
	return desc;
}

ID3D12GraphicsCommandList * D3DSystem::GetCommandList()
{
	return m_commandList;
}

bool D3DSystem::Reset()
{
	//WaitForSingleObjectEx(m_swapChainEvent, 100, FALSE);
	WaitForPreviousFrame();
	HRESULT hr;
	hr = m_commandAllocator[m_frameIndex]->Reset();
	if (FAILED(hr))
	{
		return false;
	}
	hr = m_commandList->Reset(m_commandAllocator[m_frameIndex], NULL);
	if (FAILED(hr))
	{
		return false;
	}
	//PIXBeginEvent(m_commandQueue, 0, L"Render");
	return true;
}

bool D3DSystem::Execute()
{
	m_commandList->Close();
	ID3D12CommandList* ppCommandLists[] = { m_commandList };
	m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
	HRESULT hr = m_commandQueue->Signal(m_fence[m_frameIndex], m_fenceValue[m_frameIndex]);
	if (FAILED(hr))
	{
		return false;
	}
	return true;
}

bool D3DSystem::ExecuteGraphics()
{
	m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));
	m_commandList->Close();
	ID3D12CommandList* ppCommandLists[] = { m_commandList };
	m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
	HRESULT hr = m_commandQueue->Signal(m_fence[m_frameIndex], m_fenceValue[m_frameIndex]);
	if (FAILED(hr))
	{
		return false;
	}
	return true;
}

void D3DSystem::UpdatePipelineAndClear(Vector3 Bg)
{
	m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex, m_rtvDescriptorSize);
	CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(m_dsDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
	const float clearColor[] = { Bg.x, Bg.y, Bg.z, 1.0f };
	m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	m_commandList->ClearDepthStencilView(m_dsDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
}

void D3DSystem::Wait()
{
	WaitForPreviousFrame();
	HRESULT hr = m_commandQueue->Signal(m_fence[m_frameIndex], m_fenceValue[m_frameIndex]);
}

bool D3DSystem::PresentSimple()
{
	HRESULT hr;
	// present the current backbuffer
	hr = m_swapChain->Present(0, 0);
	//hr = m_swapChain->Present(1, 0);
	if (FAILED(hr))
	{
		return false;
	}
	return true;
}

void D3DSystem::Cleanup()
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

	SAFE_RELEASE(m_device);
	SAFE_RELEASE(m_swapChain);
	SAFE_RELEASE(m_commandQueue);
	SAFE_RELEASE(m_rtvDescriptorHeap);
	SAFE_RELEASE(m_commandList);
	SAFE_RELEASE(m_depthStencilBuffer);
	SAFE_RELEASE(m_dsDescriptorHeap);

	for (int i = 0; i < FRAMEBUFFERCOUNT; ++i)
	{
		SAFE_RELEASE(m_renderTargets[i]);
		SAFE_RELEASE(m_commandAllocator[i]);
		SAFE_RELEASE(m_fence[i]);
	};
}

int D3DSystem::GetFrameIndex()
{
	return m_swapChain->GetCurrentBackBufferIndex();
}

D3D12_FEATURE_DATA_ROOT_SIGNATURE D3DSystem::GetFeatureData()
{
	return m_featureData;
}

bool D3DSystem::WaitForPreviousFrame()
{
	HRESULT hr;
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
	if (m_fence[m_frameIndex]->GetCompletedValue() < m_fenceValue[m_frameIndex])
	{
		hr = m_fence[m_frameIndex]->SetEventOnCompletion(m_fenceValue[m_frameIndex], m_fenceEvent);
		if (FAILED(hr))
		{
			return false;
		}
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}
	m_fenceValue[m_frameIndex]++;
	return true;
}
