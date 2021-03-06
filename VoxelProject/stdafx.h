// stdafx.h: включаемый файл для стандартных системных включаемых файлов
// или включаемых файлов для конкретного проекта, которые часто используются, но
// не часто изменяются
//

#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Исключите редко используемые компоненты из заголовков Windows
// Файлы заголовков Windows:
#include <windows.h>
////
#include <d3d12.h>
#include <dxgi1_4.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include "d3dx12.h"
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <ctime>
//#include <wrl.h>
//#include <shellapi.h>
////
#include "Keyboard.h"
#include "Mouse.h"
#include "SimpleMath.h"
////
#include "vector_types.h"

// Файлы заголовков C RunTime
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <tchar.h>



//#define SAFE_RELEASE(p) { if ( (p) ) { (p)->Release(); (p) = 0; } }

using namespace std;
using namespace DirectX;
using namespace DirectX::SimpleMath;
//using Microsoft::WRL::ComPtr;

inline void ThrowIfFailed(HRESULT hr)
{
	if (FAILED(hr))
	{
		throw std::exception();
	}
}
// TODO: Установите здесь ссылки на дополнительные заголовки, требующиеся для программы
