// VoxelProject.cpp: Определяет точку входа для приложения.
//

#include "stdafx.h"
#include "VoxelProject.h"

#include "GeneralModel.h"
#include "InputController.h"


#define MAX_LOADSTRING 100

// Глобальные переменные:
HINSTANCE hInst;                                // текущий экземпляр
WCHAR szTitle[MAX_LOADSTRING];                  // Текст строки заголовка
WCHAR szWindowClass[MAX_LOADSTRING];            // имя класса главного окна
WCHAR szWindowDirectxClass[MAX_LOADSTRING];            // имя класса Directx окна
WCHAR text[MAX_LOADSTRING];       //Для получения текста

int width = 1024;
int height = 768;
float PanelSize = 0.3f;
HWND hWnd;
HWND hWndDirectx;
bool isRunning = true;
string Title = "Directx Voxel Project. FPS: ";
int T = 0;

shared_ptr<GeneralModel> generalModel;
shared_ptr<InputController> inputController;


void MainLoop();


// Отправить объявления функций, включенных в этот модуль кода:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK    MainWndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: разместите код здесь.

    // Инициализация глобальных строк
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_VOXELPROJECT, szWindowClass, MAX_LOADSTRING);
	LoadStringW(hInstance, IDC_VOXELPROJECTD3DWIN, szWindowDirectxClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Выполнить инициализацию приложения:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_VOXELPROJECT));

	generalModel = make_shared<GeneralModel>(hWndDirectx, width, height);
	inputController = make_shared<InputController>(generalModel);

	MainLoop();
	//generalModel.CleanUp();

	/*
    MSG msg;

    // Цикл основного сообщения:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
	*/
	return 0;
}



//
//  ФУНКЦИЯ: MyRegisterClass()
//
//  НАЗНАЧЕНИЕ: регистрирует класс окна.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEXW wcex;

	wcex.cbSize = sizeof(WNDCLASSEX);

	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = MainWndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_VOXELPROJECT));
	wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_VOXELPROJECT);
	wcex.lpszClassName = szWindowClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

	RegisterClassExW(&wcex);

	WNDCLASSEXW wcexChild;

	wcexChild.cbSize = sizeof(WNDCLASSEX);

	wcexChild.style = CS_HREDRAW | CS_VREDRAW;
	wcexChild.lpfnWndProc = WndProc;
	wcexChild.cbClsExtra = 0;
	wcexChild.cbWndExtra = 0;
	wcexChild.hInstance = hInstance;
	wcexChild.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_VOXELPROJECT));
	wcexChild.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcexChild.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcexChild.lpszMenuName = NULL;
	wcexChild.lpszClassName = szWindowDirectxClass;
	wcexChild.hIconSm = LoadIcon(wcexChild.hInstance, MAKEINTRESOURCE(IDI_SMALL));

	return RegisterClassExW(&wcexChild);
}

//
//   ФУНКЦИЯ: InitInstance(HINSTANCE, int)
//
//   НАЗНАЧЕНИЕ: сохраняет обработку экземпляра и создает главное окно.
//
//   КОММЕНТАРИИ:
//
//        В данной функции дескриптор экземпляра сохраняется в глобальной переменной, а также
//        создается и выводится на экран главное окно программы.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	hInst = hInstance; // Сохранить дескриптор экземпляра в глобальной переменной

	hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, 0, width + PanelSize * width, height, nullptr, nullptr, hInstance, nullptr);

	hWndDirectx = CreateWindowW(szWindowDirectxClass, TEXT("Direct window"), WS_CHILD,
		CW_USEDEFAULT, 0, width, height, hWnd, nullptr, hInstance, nullptr);

	if (!hWnd)
	{
		return FALSE;
	}

	if (!hWndDirectx)
	{
		return FALSE;
	}

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	ShowWindow(hWndDirectx, nCmdShow);
	UpdateWindow(hWndDirectx);

	return TRUE;
}

//
//  ФУНКЦИЯ: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  НАЗНАЧЕНИЕ:  обрабатывает сообщения в главном окне.
//
//  WM_COMMAND — обработать меню приложения
//  WM_PAINT — отрисовать главное окно
//  WM_DESTROY — отправить сообщение о выходе и вернуться
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
		
	case WM_ACTIVATEAPP:
		Keyboard::ProcessMessage(message, wParam, lParam);
		Mouse::ProcessMessage(message, wParam, lParam);
		break;

	case WM_LBUTTONDOWN:
		SetFocus(hWndDirectx);
	case WM_INPUT:
	case WM_MOUSEMOVE:
	case WM_LBUTTONUP:
	case WM_RBUTTONDOWN:
	case WM_RBUTTONUP:
	case WM_MBUTTONDOWN:
	case WM_MBUTTONUP:
	case WM_MOUSEWHEEL:
	case WM_XBUTTONDOWN:
	case WM_XBUTTONUP:
	case WM_MOUSEHOVER:
		Mouse::ProcessMessage(message, wParam, lParam);
		break;

	case WM_KEYDOWN:
	case WM_SYSKEYDOWN:
	case WM_KEYUP:
	case WM_SYSKEYUP:
		Keyboard::ProcessMessage(message, wParam, lParam);
		break;

	case WM_COMMAND:
		{
			int wmId = LOWORD(wParam);
			// Разобрать выбор в меню:
			switch (wmId)
			{
			case IDM_ABOUT:
				DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
				break;
			case IDM_EXIT:
				isRunning = false;
				DestroyWindow(hWnd);
				break;
			default:
				return DefWindowProc(hWnd, message, wParam, lParam);
			}
		}
		break;
	case WM_PAINT:
		{
			PAINTSTRUCT ps;
			HDC hdc = BeginPaint(hWnd, &ps);
			// TODO: Добавьте сюда любой код прорисовки, использующий HDC...
			EndPaint(hWnd, &ps);
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

LRESULT CALLBACK MainWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{	
	case WM_CREATE:
		break;
	case WM_CTLCOLORSTATIC:
		return (BOOL)GetSysColorBrush(COLOR_WINDOW);
		break;
	case WM_LBUTTONDOWN:
		SetFocus(hWnd);
		break;
	case WM_COMMAND:
	{
		int wmId = LOWORD(wParam);
		// Разобрать выбор в меню:
		switch (wmId)
		{
		case IDM_ABOUT:
			DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
			break;
		case IDM_EXIT:
			isRunning = false;
			DestroyWindow(hWnd);
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
	}
	break;
	case WM_PAINT:
	{
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hWnd, &ps);
		// TODO: Добавьте сюда любой код прорисовки, использующий HDC...
		EndPaint(hWnd, &ps);
	}
	break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

// Обработчик сообщений для окна "О программе".
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}

void MainLoop()
{
	MSG msg;
	while (isRunning)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				break;

			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else {
			inputController->Update();
			generalModel->Render();
			//T = clock();
			//inputController.Update();
			//generalModel.Update();
			//generalModel.Render();
			//T = (clock() - T);
			/*
			if (T == 0)
			{
				SetWindowTextA(hWnd, (Title + "inf").c_str());
			}
			else
			{
				SetWindowTextA(hWnd, (Title + to_string(CLOCKS_PER_SEC / T)).c_str());
			}
			*/
		}
	}
}
