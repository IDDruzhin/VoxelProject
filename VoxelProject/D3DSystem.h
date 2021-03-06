#pragma once

#define FRAMEBUFFERCOUNT 3

#include "Structures.h"
#include <wrl.h>
#include <shellapi.h>

using Microsoft::WRL::ComPtr;

class D3DSystem
{
public:
	D3DSystem(HWND hWnd, int width, int height);
	~D3DSystem();
	ComPtr<ID3D12Device> GetDevice();
	DXGI_SWAP_CHAIN_DESC GetSwapChainDesc();
	ComPtr<ID3D12GraphicsCommandList> GetCommandList();
	void Reset();
	void Execute();
	void UpdatePipeline();
	void Wait();
	void PresentSimple();
	ID3D12Resource* GetRenderTarget();
	CD3DX12_CPU_DESCRIPTOR_HANDLE GetRtvCPUHandle();
	template<typename T>
	ComPtr<ID3D12Resource> CreateDefaultBuffer(T* data, int size, D3D12_RESOURCE_STATES finalState, D3D12_RESOURCE_DESC desc, wstring name = L"");
	template<typename T>
	ComPtr<ID3D12Resource> CreateVertexBuffer(T* data, int size, wstring name = L"");
	template<typename T>
	ComPtr<ID3D12Resource> CreateIndexBuffer(T* data, int size, wstring name = L"");
	template<typename T>
	ComPtr<ID3D12Resource> CreateStructuredBuffer(T* data, int size, wstring name = L"");
	template<typename T>
	ComPtr<ID3D12Resource> CreateRWStructuredBuffer(T* data, int size, wstring name = L"");
	template<typename T>
	void CopyDataFromGPU(ComPtr<ID3D12Resource> src, T * dst, int size);
	template<typename T>
	void CopyDataToGPU(ComPtr<ID3D12Resource> dst, T * data, int size);
	template<typename T>
	void CopyDataToGPU(ComPtr<ID3D12Resource> dst, T * data, int size, ComPtr<ID3D12Resource> uploadBuffer);
	ComPtr<ID3D12Resource> CreateRWTexture3D(int3 dim, DXGI_FORMAT format, wstring name);
	template<typename T>
	ComPtr<ID3D12Resource> CreateTexture1D(vector<T>& data, DXGI_FORMAT format, wstring name);
	int GetFrameIndex();
	D3D12_FEATURE_DATA_ROOT_SIGNATURE GetFeatureData();
private:
	ComPtr<ID3D12Device> m_device;
	ComPtr<IDXGISwapChain3> m_swapChain;
	ComPtr<ID3D12CommandQueue> m_commandQueue;
	ComPtr<ID3D12DescriptorHeap> m_rtvDescriptorHeap;
	ComPtr<ID3D12Resource> m_renderTargets[FRAMEBUFFERCOUNT];
	ComPtr<ID3D12CommandAllocator> m_commandAllocator[FRAMEBUFFERCOUNT];
	ComPtr<ID3D12GraphicsCommandList> m_commandList;
	ComPtr<ID3D12Fence> m_fence[FRAMEBUFFERCOUNT];
	HANDLE m_fenceEvent;
	UINT64 m_fenceValue[FRAMEBUFFERCOUNT];
	int m_frameIndex;
	int m_rtvDescriptorSize;
	ComPtr<ID3D12Resource> m_depthStencilBuffer;
	ComPtr<ID3D12DescriptorHeap> m_dsDescriptorHeap;
	D3D12_FEATURE_DATA_ROOT_SIGNATURE m_featureData;
	void OnDestroy();
};


template<typename T>
inline ComPtr<ID3D12Resource> D3DSystem::CreateDefaultBuffer(T * data, int size, D3D12_RESOURCE_STATES finalState, D3D12_RESOURCE_DESC desc, wstring name)
{
	ComPtr<ID3D12Resource> buffer;
	ThrowIfFailed(m_device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), //A default heap
		D3D12_HEAP_FLAG_NONE, //No flags
		&desc, //Size of buffer
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&buffer)));
	buffer->SetName(name.c_str());

	ComPtr<ID3D12Resource> bufferUploadHeap;
	ThrowIfFailed(m_device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(size),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&bufferUploadHeap)));
	bufferUploadHeap->SetName(L"Upload heap");
	D3D12_SUBRESOURCE_DATA subResourceData = {};
	subResourceData.pData = reinterpret_cast<BYTE*>(data);
	subResourceData.RowPitch = size;
	subResourceData.SlicePitch = size;

	//m_commandList
	Reset();
	UpdateSubresources(m_commandList.Get(), buffer.Get(), bufferUploadHeap.Get(), 0, 0, 1, &subResourceData);
	m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, finalState));
	m_commandList->Close();
	ID3D12CommandList* commandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	m_fenceValue[m_frameIndex]++;
	m_commandQueue->Signal(m_fence[m_frameIndex].Get(), m_fenceValue[m_frameIndex]);
	if (m_fence[m_frameIndex]->GetCompletedValue() < m_fenceValue[m_frameIndex])
	{
		m_fence[m_frameIndex]->SetEventOnCompletion(m_fenceValue[m_frameIndex], m_fenceEvent);
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}
	return buffer;
}

template<typename T>
inline ComPtr<ID3D12Resource> D3DSystem::CreateVertexBuffer(T * data, int size, wstring name)
{
	D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(size);
	return CreateDefaultBuffer(data, size, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, desc, name);
}

template<typename T>
inline ComPtr<ID3D12Resource> D3DSystem::CreateIndexBuffer(T * data, int size, wstring name)
{
	D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(size);
	return CreateDefaultBuffer(data, size, D3D12_RESOURCE_STATE_INDEX_BUFFER, desc, name);
}

template<typename T>
inline ComPtr<ID3D12Resource> D3DSystem::CreateStructuredBuffer(T * data, int size, wstring name)
{

	D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	return CreateDefaultBuffer(data, size, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, desc, name);
}

template<typename T>
inline ComPtr<ID3D12Resource> D3DSystem::CreateRWStructuredBuffer(T * data, int size, wstring name)
{

	D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	return CreateDefaultBuffer(data, size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, desc, name);
}

inline ComPtr<ID3D12Resource> D3DSystem::CreateRWTexture3D(int3 dim, DXGI_FORMAT format, wstring name)
{
	CD3DX12_RESOURCE_DESC textureDesc = CD3DX12_RESOURCE_DESC::Tex3D(format, dim.x, dim.y, dim.z, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	ComPtr<ID3D12Resource> buffer;
	ThrowIfFailed(m_device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&textureDesc,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		nullptr,
		IID_PPV_ARGS(&buffer)));
	return buffer;
}

template<typename T>
inline ComPtr<ID3D12Resource> D3DSystem::CreateTexture1D(vector<T>& data, DXGI_FORMAT format, wstring name)
{
	CD3DX12_RESOURCE_DESC textureDesc = CD3DX12_RESOURCE_DESC::Tex1D(format, data.size());
	return CreateDefaultBuffer(&data[0], sizeof(T)*data.size(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, textureDesc, name);
}

template<typename T>
inline void D3DSystem::CopyDataFromGPU(ComPtr<ID3D12Resource> src, T * dst, int size)
{
	ComPtr<ID3D12Resource> buffer;
	m_device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(size),
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&buffer));
	buffer->SetName(L"Readback heap");

	//m_commandList
	Reset();
	m_commandList->CopyResource(buffer.Get(), src.Get());
	m_commandList->Close();
	ID3D12CommandList* commandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	m_fenceValue[m_frameIndex]++;
	m_commandQueue->Signal(m_fence[m_frameIndex].Get(), m_fenceValue[m_frameIndex]);
	if (m_fence[m_frameIndex]->GetCompletedValue() < m_fenceValue[m_frameIndex])
	{
		m_fence[m_frameIndex]->SetEventOnCompletion(m_fenceValue[m_frameIndex], m_fenceEvent);
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}
	T* buffData;
	buffer->Map(0, nullptr, reinterpret_cast<void**>(&buffData));
	memcpy(dst, buffData, size);
	buffer->Unmap(0, nullptr);
}

template<typename T>
inline void D3DSystem::CopyDataToGPU(ComPtr<ID3D12Resource> dst, T * data, int size)
{

	ComPtr<ID3D12Resource> buffer;
	m_device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(size),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&buffer));
	buffer->SetName(L"Upload heap");
	T* buffData;
	buffer->Map(0, nullptr, reinterpret_cast<void**>(&buffData));
	memcpy(buffData, data, size);
	buffer->Unmap(0, nullptr);

	//m_commandList
	Reset();
	m_commandList->CopyResource(dst.Get(), buffer.Get());
	m_commandList->Close();
	ID3D12CommandList* commandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	m_fenceValue[m_frameIndex]++;
	m_commandQueue->Signal(m_fence[m_frameIndex].Get(), m_fenceValue[m_frameIndex]);
	if (m_fence[m_frameIndex]->GetCompletedValue() < m_fenceValue[m_frameIndex])
	{
		m_fence[m_frameIndex]->SetEventOnCompletion(m_fenceValue[m_frameIndex], m_fenceEvent);
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}
}

template<typename T>
inline void D3DSystem::CopyDataToGPU(ComPtr<ID3D12Resource> dst, T * data, int size, ComPtr<ID3D12Resource> uploadBuffer)
{
	D3D12_SUBRESOURCE_DATA subResourceData = {};
	subResourceData.pData = reinterpret_cast<BYTE*>(data);
	subResourceData.RowPitch = size;
	subResourceData.SlicePitch = size;

	//m_commandList
	Reset();
	UpdateSubresources(m_commandList.Get(), dst.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);
	m_commandList->Close();
	ID3D12CommandList* commandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	m_fenceValue[m_frameIndex]++;
	m_commandQueue->Signal(m_fence[m_frameIndex].Get(), m_fenceValue[m_frameIndex]);
	if (m_fence[m_frameIndex]->GetCompletedValue() < m_fenceValue[m_frameIndex])
	{
		m_fence[m_frameIndex]->SetEventOnCompletion(m_fenceValue[m_frameIndex], m_fenceEvent);
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}
}
