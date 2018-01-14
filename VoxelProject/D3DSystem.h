#pragma once

#include "Structures.h"

#define FRAMEBUFFERCOUNT 3

class D3DSystem
{
public:
	D3DSystem();
	~D3DSystem();
	bool InitD3D(HWND hWnd, int Width, int Height);
	ID3D12Device* GetDevice();
	DXGI_SWAP_CHAIN_DESC GetSwapChainDesc();
	ID3D12GraphicsCommandList* GetCommandList();
	bool Reset();
	bool Execute();
	bool ExecuteGraphics();
	void UpdatePipelineAndClear(Vector3 Bg);
	void Wait();
	bool PresentSimple();
	template<typename T>
	ID3D12Resource * CreateDefaultBuffer(T* Data, int Size, D3D12_RESOURCE_STATES FinalState, D3D12_RESOURCE_DESC Desc, wstring Name = L"");
	template<typename T>
	ID3D12Resource * CreateVertexBuffer(T* Data, int Size, wstring Name = L"");
	template<typename T>
	ID3D12Resource * CreateStructuredBuffer(T* Data, int Size, wstring Name = L"");
	template<typename T>
	void CopyDataFromGPU(ID3D12Resource * Src, T * Dst, int Size);
	template<typename T>
	void CopyDataToGPU(ID3D12Resource * Src, T * Dst, int Size);
	void Cleanup();
	int GetFrameIndex();
	D3D12_FEATURE_DATA_ROOT_SIGNATURE GetFeatureData();
private:

	ID3D12Device * device;
	IDXGISwapChain3* swapChain;
	ID3D12CommandQueue* commandQueue;
	ID3D12DescriptorHeap* rtvDescriptorHeap;
	ID3D12Resource* renderTargets[FRAMEBUFFERCOUNT];
	ID3D12CommandAllocator* commandAllocator[FRAMEBUFFERCOUNT];
	ID3D12GraphicsCommandList* commandList;
	ID3D12Fence* fence[FRAMEBUFFERCOUNT];
	HANDLE fenceEvent;
	UINT64 fenceValue[FRAMEBUFFERCOUNT];
	int frameIndex;
	int rtvDescriptorSize;

	ID3D12Resource* depthStencilBuffer;
	ID3D12DescriptorHeap* dsDescriptorHeap;

	bool WaitForPreviousFrame();

	D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData;

	HANDLE swapChainEvent;
};


template<typename T>
inline ID3D12Resource * D3DSystem::CreateDefaultBuffer(T * Data, int Size, D3D12_RESOURCE_STATES FinalState, D3D12_RESOURCE_DESC Desc, wstring Name)
{
	ID3D12Resource * Buffer;
	device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), //A default heap
		D3D12_HEAP_FLAG_NONE, //No flags
		&Desc, //Size of buffer
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&Buffer));
	Buffer->SetName(Name.c_str());

	ID3D12Resource* BufferUploadHeap;
	device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), //An upload heap
		D3D12_HEAP_FLAG_NONE, //No flags
		&CD3DX12_RESOURCE_DESC::Buffer(Size),
		D3D12_RESOURCE_STATE_GENERIC_READ, //GPU will read and copy content to the default heap
		nullptr,
		IID_PPV_ARGS(&BufferUploadHeap));
	BufferUploadHeap->SetName(L"Buffer Upload Resource Heap");
	D3D12_SUBRESOURCE_DATA SubResourceData = {};
	SubResourceData.pData = reinterpret_cast<BYTE*>(Data); //Pointer to upload data
	SubResourceData.RowPitch = Size;
	SubResourceData.SlicePitch = Size;

	int CurF = fence[frameIndex]->GetCompletedValue();
	Reset();
	CurF = fence[frameIndex]->GetCompletedValue();
	UpdateSubresources(commandList, Buffer, BufferUploadHeap, 0, 0, 1, &SubResourceData);
	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(Buffer, D3D12_RESOURCE_STATE_COPY_DEST, FinalState));
	commandList->Close();
	//commandList
	ID3D12CommandList* CommandLists[] = { commandList };
	commandQueue->ExecuteCommandLists(_countof(CommandLists), CommandLists);
	commandQueue->Signal(fence[frameIndex], fenceValue[frameIndex]);
	CurF = fence[frameIndex]->GetCompletedValue();
	if (fence[frameIndex]->GetCompletedValue() < fenceValue[frameIndex])
	{
		fence[frameIndex]->SetEventOnCompletion(fenceValue[frameIndex], fenceEvent);
		WaitForSingleObject(fenceEvent, INFINITE);
	}
	CurF = fence[frameIndex]->GetCompletedValue();
	BufferUploadHeap->Release();
	return Buffer;
}

template<typename T>
inline ID3D12Resource * D3DSystem::CreateVertexBuffer(T * Data, int Size, wstring Name)
{
	D3D12_RESOURCE_DESC Desc = CD3DX12_RESOURCE_DESC::Buffer(Size);
	return CreateDefaultBuffer(Data, Size, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, Desc, Name);
}

template<typename T>
inline ID3D12Resource * D3DSystem::CreateStructuredBuffer(T * Data, int Size, wstring Name)
{

	D3D12_RESOURCE_DESC Desc = CD3DX12_RESOURCE_DESC::Buffer(Size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	//return CreateDefaultBuffer(Data, Size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, Desc, Name);
	return CreateDefaultBuffer(Data, Size, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, Desc, Name);
}

template<typename T>
inline void D3DSystem::CopyDataFromGPU(ID3D12Resource * Src, T * Dst, int Size)
{
	ID3D12Resource * Buffer;
	device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK), //A readback heap
		D3D12_HEAP_FLAG_NONE, //No flags
		&CD3DX12_RESOURCE_DESC::Buffer(Size), //Size of buffer
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&Buffer));
	Buffer->SetName(L"Buffer Readback Resource Heap");
	//commandList
	Reset();
	commandList->CopyResource(Buffer, Src);
	commandList->Close();
	ID3D12CommandList* CommandLists[] = { commandList };
	commandQueue->ExecuteCommandLists(_countof(CommandLists), CommandLists);
	commandQueue->Signal(fence[frameIndex], fenceValue[frameIndex]);
	if (fence[frameIndex]->GetCompletedValue() < fenceValue[frameIndex])
	{
		fence[frameIndex]->SetEventOnCompletion(fenceValue[frameIndex], fenceEvent);
		WaitForSingleObject(fenceEvent, INFINITE);
	}
	T* pBuffData;
	Buffer->Map(0, nullptr, reinterpret_cast<void**>(&pBuffData));
	memcpy(Dst, pBuffData, Size);
	Buffer->Unmap(0, nullptr);
	Buffer->Release();
}

template<typename T>
inline void D3DSystem::CopyDataToGPU(ID3D12Resource * Dst, T * Data, int Size)
{

	ID3D12Resource* Buffer;
	device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), //An upload heap
		D3D12_HEAP_FLAG_NONE, //No flags
		&CD3DX12_RESOURCE_DESC::Buffer(Size),
		D3D12_RESOURCE_STATE_GENERIC_READ, //GPU will read and copy content to the default heap
		nullptr,
		IID_PPV_ARGS(&Buffer));
	Buffer->SetName(L"Buffer for upload");
	T* pBuffData;
	Buffer->Map(0, nullptr, reinterpret_cast<void**>(&pBuffData));
	memcpy(pBuffData, Data, Size);
	Buffer->Unmap(0, nullptr);
	//commandList
	Reset();
	commandList->CopyResource(Dst, Buffer);
	commandList->Close();
	ID3D12CommandList* CommandLists[] = { commandList };
	commandQueue->ExecuteCommandLists(_countof(CommandLists), CommandLists);
	commandQueue->Signal(fence[frameIndex], fenceValue[frameIndex]);
	if (fence[frameIndex]->GetCompletedValue() < fenceValue[frameIndex])
	{
		fence[frameIndex]->SetEventOnCompletion(fenceValue[frameIndex], fenceEvent);
		WaitForSingleObject(fenceEvent, INFINITE);
	}
	Buffer->Release();
}
