// VoxelProject.cpp: Определяет точку входа для приложения.
//

#include "stdafx.h"
#include "VoxelProject.h"

#include "GeneralModel.h"
#include "InputController.h"
#include "Commctrl.h"
#include "Commdlg.h"
#include "Windowsx.h"
//#include "atlstr.h"


#define MAX_LOADSTRING 100

// Глобальные переменные:
HINSTANCE hInst;                                // текущий экземпляр
WCHAR szTitle[MAX_LOADSTRING];                  // Текст строки заголовка
WCHAR szWindowClass[MAX_LOADSTRING];            // имя класса главного окна
WCHAR szWindowDirectxClass[MAX_LOADSTRING];            // имя класса Directx окна
WCHAR text[1000];       //Для получения текста

int width = 1024;
int height = 768;
float PanelSize = 0.3f;
HWND hWndDirectx;
HWND hWndRayCastingDlg;
HWND hWndSkeletalDlg;
HWND hStepSize;
HWND hBlockSize;
HWND hLinerp;
HWND hShowBlocks;
HWND hSegmentsList;
HWND hFPS;
HWND hMeanFPS;
HWND hBoneLength;
HWND hXRotate;
HWND hYRotate;
HWND hZRotate;
HWND hXTranslate;
HWND hYTranslate;
HWND hZTranslate;
HWND hBorderSegment;
HWND hBindBonesButton;
HWND hAddBonesButton;
HWND hDeleteBonesButton;
HWND hCopyBonesButton;
HWND hBonesThickness;
bool isRunning = true;
bool isLoaded = false;
string Title = "Voxel Project. FPS: ";
int T = 0;
vector<float> segmentsOpacity;
uint totalTime;
uint framesCount;

shared_ptr<GeneralModel> generalModel;
shared_ptr<InputController> inputController;


void MainLoop();


// Отправить объявления функций, включенных в этот модуль кода:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    DirectxWndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK RayCastingDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
INT_PTR CALLBACK SkeletalDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

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

	totalTime = 0;
	framesCount = 0;

	MainLoop();

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
	wcex.lpfnWndProc = DirectxWndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_VOXELPROJECT));
	wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_VOXELPROJECT);
	wcex.lpszClassName = szWindowDirectxClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

	return RegisterClassExW(&wcex);

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

	
	RECT rect;
	rect.left = 0;
	rect.right = width;
	rect.bottom = height;
	rect.top = 0;
	AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
	hWndDirectx = CreateWindowW(szWindowDirectxClass, szTitle, WS_OVERLAPPEDWINDOW,
		0, 0, rect.right - rect.left, rect.bottom - rect.top, nullptr, nullptr, hInstance, nullptr);

	if (!hWndDirectx)
	{
		return FALSE;
	}

	ShowWindow(hWndDirectx, nCmdShow);
	UpdateWindow(hWndDirectx);

	hWndRayCastingDlg = CreateDialogParam(hInstance, MAKEINTRESOURCE(IDD_RC_DIALOG), 0, (RayCastingDlgProc), 0);
	ShowWindow(hWndRayCastingDlg, nCmdShow);
	UpdateWindow(hWndRayCastingDlg);

	hWndSkeletalDlg = CreateDialogParam(hInstance, MAKEINTRESOURCE(IDD_SKELETAL_DIALOG), 0, (SkeletalDlgProc), 0);
	ShowWindow(hWndSkeletalDlg, nCmdShow);
	UpdateWindow(hWndSkeletalDlg);

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
LRESULT CALLBACK DirectxWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
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
			case ID_FILE_LOADBIN:
				{
					WCHAR szFile[1000];
					OPENFILENAME ofn;
					ZeroMemory(&ofn, sizeof(ofn));
					ofn.lStructSize = sizeof(ofn);
					ofn.hwndOwner = NULL;
					ofn.lpstrFile = szFile;
					ofn.lpstrFile[0] = '\0';
					ofn.nMaxFile = sizeof(szFile);
					ofn.lpstrFilter = L"All\0*.*\0BIN\0*.bin\0";
					ofn.nFilterIndex = 1;
					ofn.lpstrFileTitle = NULL;
					ofn.nMaxFileTitle = 0;
					ofn.lpstrInitialDir = NULL;
					ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
					if (GetOpenFileName(&ofn))
					{
						string path(&szFile[0], &szFile[999]);
						generalModel->LoadBin(path);
						GetWindowText(hBlockSize, text, 1000);
						generalModel->BlocksDecomposition(wcstof(text, NULL));
						GetWindowText(hStepSize, text, 1000);
						generalModel->SetStepSize(wcstof(text, NULL));

						vector<string> segmentsNames = generalModel->GetObjectSegmentsNames();
						segmentsOpacity = generalModel->GetSegmentsOpacity();

						ListView_DeleteAllItems(hSegmentsList);
						LVITEM item;
						item.mask = LVIF_TEXT;
						item.iSubItem = 0;
						item.iItem = 0;
						for (int i = segmentsNames.size() - 1; i >= 0; i--)
						{
							wstring name(segmentsNames[i].begin(), segmentsNames[i].end());
							wstring sOpacity = to_wstring(segmentsOpacity[i]);
							wstring sID = to_wstring(i);
							item.pszText = (LPWSTR)(sOpacity.c_str());
							ListView_InsertItem(hSegmentsList, &item);
							ListView_SetItemText(hSegmentsList, 0, 1, (LPWSTR)(sID.c_str()));
							ListView_SetItemText(hSegmentsList, 0, 2, (LPWSTR)(name.c_str()));
						}
						if (generalModel->IsBonesBinded())
						{
							SendMessage(hBindBonesButton, WM_SETTEXT, 0, (LPARAM)L"Unbind");
							EnableWindow(hAddBonesButton, FALSE);
							EnableWindow(hDeleteBonesButton, FALSE);
							EnableWindow(hCopyBonesButton, FALSE);
						}
					}
					totalTime = 0;
					framesCount = 0;
					isLoaded = true;
				}
				break;
			case ID_FILE_LOADFROMIMAGES:
				{
					WCHAR szFile[1000];
					OPENFILENAME ofn;
					ZeroMemory(&ofn, sizeof(ofn));
					ofn.lStructSize = sizeof(ofn);
					ofn.hwndOwner = NULL;
					ofn.lpstrFile = szFile;
					ofn.lpstrFile[0] = '\0';
					ofn.nMaxFile = sizeof(szFile);
					ofn.lpstrFilter = L"All\0*.*\0Text\0*.txt\0";
					ofn.nFilterIndex = 1;
					ofn.lpstrFileTitle = NULL;
					ofn.nMaxFileTitle = 0;
					ofn.lpstrInitialDir = NULL;
					ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
					if (GetOpenFileName(&ofn))
					{
						string path(&szFile[0], &szFile[999]);
						generalModel->LoadFromImages(path);
						GetWindowText(hBlockSize, text, 1000);
						generalModel->BlocksDecomposition(wcstof(text, NULL));
						GetWindowText(hStepSize, text, 1000);
						generalModel->SetStepSize(wcstof(text, NULL));

						vector<string> segmentsNames = generalModel->GetObjectSegmentsNames();
						segmentsOpacity = generalModel->GetSegmentsOpacity();

						ListView_DeleteAllItems(hSegmentsList);
						LVITEM item;
						item.mask = LVIF_TEXT;
						item.iSubItem = 0;
						item.iItem = 0;
						for (int i = segmentsNames.size() - 1; i >= 0; i--)
						{
							wstring name(segmentsNames[i].begin(), segmentsNames[i].end());
							wstring sOpacity = to_wstring(segmentsOpacity[i]);
							wstring sID = to_wstring(i);
							item.pszText = (LPWSTR)(sOpacity.c_str());
							ListView_InsertItem(hSegmentsList, &item);
							ListView_SetItemText(hSegmentsList, 0, 1, (LPWSTR)(sID.c_str()));
							ListView_SetItemText(hSegmentsList, 0, 2, (LPWSTR)(name.c_str()));
						}
						SendMessage(hBindBonesButton, WM_SETTEXT, 0, (LPARAM)L"Bind");
						EnableWindow(hAddBonesButton, TRUE);
						EnableWindow(hDeleteBonesButton, TRUE);
						EnableWindow(hCopyBonesButton, TRUE);
					}
					totalTime = 0;
					framesCount = 0;
					isLoaded = true;
				}
				break;
			case ID_FILE_SAVEBIN:
				{
					WCHAR szFile[1000];
					OPENFILENAME ofn;
					ZeroMemory(&ofn, sizeof(ofn));
					ofn.lStructSize = sizeof(ofn);
					ofn.hwndOwner = NULL;
					ofn.lpstrFile = szFile;
					ofn.lpstrFile[0] = '\0';
					ofn.nMaxFile = sizeof(szFile);
					ofn.lpstrFilter = L"BIN\0*.bin\0";
					ofn.nFilterIndex = 1;
					ofn.lpstrFileTitle = NULL;
					ofn.nMaxFileTitle = 0;
					ofn.lpstrInitialDir = NULL;
					ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
					if (GetSaveFileName(&ofn))
					{
						string path(&szFile[0], &szFile[999]);
						int end = path.find('\0');
						path.resize(end);
						generalModel->SaveBin(path);
					}
					totalTime = 0;
					framesCount = 0;
				}
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
		case WM_SIZING:
		{
			RECT rc;
			GetClientRect(hWnd, &rc);
			Vector2 clientSize(rc.right - rc.left, rc.bottom - rc.top);
			generalModel->SetClientSize(clientSize);
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

INT_PTR CALLBACK RayCastingDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_INITDIALOG:
	{
		SetWindowPos(hWnd, HWND_TOP, width, 0, width * PanelSize, height, 0);
		hStepSize = GetDlgItem(hWnd, IDC_STEPSIZE_EDIT);
		SendMessage(hStepSize, WM_SETTEXT, 0, (LPARAM)L"0.5");
		hBlockSize = GetDlgItem(hWnd, IDC_BLOCKSIZE_EDIT);
		SendMessage(hBlockSize, WM_SETTEXT, 0, (LPARAM)L"256");
		hLinerp = GetDlgItem(hWnd, IDC_LINTERP_CHECK);
		hShowBlocks = GetDlgItem(hWnd, IDC_SHOW_BLOCKS_CHECK);
		hSegmentsList = GetDlgItem(hWnd, IDC_SEGMENTS_LIST);
		hFPS = GetDlgItem(hWnd, IDC_FPS_EDIT);
		hMeanFPS = GetDlgItem(hWnd, IDC_MEAN_FPS_EDIT);
		LVCOLUMN col;
		col.mask = LVCF_WIDTH | LVCF_TEXT;
		col.cx = 60;
		col.pszText = (LPWSTR)L"Opacity";
		ListView_InsertColumn(hSegmentsList, 0, &col);
		col.cx = 30;
		col.pszText = (LPWSTR)L"ID";
		ListView_InsertColumn(hSegmentsList, 1, &col);
		col.cx = 150;
		col.pszText = (LPWSTR)L"Name";
		ListView_InsertColumn(hSegmentsList, 2, &col);
	}
	break;
	case WM_COMMAND:
	{
		int wmId = LOWORD(wParam);
		// Разобрать выбор в меню:
		switch (wmId)
		{
		case IDC_COMPUTE_BUTTON:
			{
				GetWindowText(hBlockSize, text, 1000);
				generalModel->BlocksDecomposition(wcstof(text, NULL));
				totalTime = 0;
				framesCount = 0;
			}
			break;
		case IDC_STEPSIZE_BUTTON:
			{
				GetWindowText(hStepSize, text, 1000);
				generalModel->SetStepSize(wcstof(text, NULL));
				totalTime = 0;
				framesCount = 0;
			}
			break;
		case IDC_SHOW_BLOCKS_CHECK:
			{
				if (SendDlgItemMessage(hWnd, IDC_SHOW_BLOCKS_CHECK, BM_GETCHECK, 0, 0))
				{
					generalModel->SetBlocksVisiblity(true);
				}
				else
				{
					generalModel->SetBlocksVisiblity(false);
				}
			}
			break;
		case IDC_LINTERP_CHECK:
			{
				if (SendDlgItemMessage(hWnd, IDC_LINTERP_CHECK, BM_GETCHECK, 0, 0))
				{
					generalModel->SetInterpolationMode(VoxelPipeline::INTERPOLATION_MODE::INTERPOLATION_MODE_TRILINEAR);
				}
				else
				{
					generalModel->SetInterpolationMode(VoxelPipeline::INTERPOLATION_MODE::INTERPOLATION_MODE_NONE);
				}
				totalTime = 0;
				framesCount = 0;
			}
			break;
		case IDC_MEAN_FPS_RESET_BUTTON:
			{
				totalTime = 0;
				framesCount = 0;
			}
			break;
		}
	}
	break;
	case WM_NOTIFY:
	{
		LPNMHDR lpNmHdr = (LPNMHDR)lParam;
		if (lpNmHdr->code == LVN_ENDLABELEDIT)
		{
			NMLVDISPINFO *lpNmlvdispInfo = (NMLVDISPINFO*)lParam;
			if (NULL != lpNmlvdispInfo->item.pszText)
			{
				float value = wcstof(lpNmlvdispInfo->item.pszText,NULL);
				if (value >= 0.0f && value <= 1.0f)
				{
					wstring sValue = to_wstring(value);
					lpNmlvdispInfo->item.pszText = (LPWSTR)(sValue.c_str());
					segmentsOpacity[lpNmlvdispInfo->item.iItem] = value;
					generalModel->SetSegmentsOpacity(segmentsOpacity);
					SetWindowLong(hWnd, 0, TRUE);
					return TRUE;
				}
			}
		}
		if (lpNmHdr->code == LVN_ITEMACTIVATE)
		{
			LPNMLISTVIEW list = (LPNMLISTVIEW)lParam;
			int index = list->iItem;
			ListView_EditLabel(hSegmentsList, index);
		}	
	}
	break;
	case WM_CLOSE:
		EndDialog(hWnd, 0);
		PostQuitMessage(0);
		return 0;
	}
	return (INT_PTR)FALSE;
}

INT_PTR CALLBACK SkeletalDlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_INITDIALOG:
	{
		SetWindowPos(hWnd, HWND_TOP, width + width * PanelSize * 0.5f , 0, width * PanelSize, height, 0);
		hBoneLength = GetDlgItem(hWnd, IDC_LENGTH_EDIT);
		SendMessage(hBoneLength, WM_SETTEXT, 0, (LPARAM)L"100");
		hXRotate = GetDlgItem(hWnd, IDC_X_ROTATE_EDIT);
		hYRotate = GetDlgItem(hWnd, IDC_Y_ROTATE_EDIT);
		hZRotate = GetDlgItem(hWnd, IDC_Z_ROTATE_EDIT);
		SendMessage(hXRotate, WM_SETTEXT, 0, (LPARAM)L"0");
		SendMessage(hYRotate, WM_SETTEXT, 0, (LPARAM)L"0");
		SendMessage(hZRotate, WM_SETTEXT, 0, (LPARAM)L"30");
		hXTranslate = GetDlgItem(hWnd, IDC_X_TRANSLATE_EDIT);
		hYTranslate = GetDlgItem(hWnd, IDC_Y_TRANSLATE_EDIT);
		hZTranslate = GetDlgItem(hWnd, IDC_Z_TRANSLATE_EDIT);
		SendMessage(hXTranslate, WM_SETTEXT, 0, (LPARAM)L"100");
		SendMessage(hYTranslate, WM_SETTEXT, 0, (LPARAM)L"0");
		SendMessage(hZTranslate, WM_SETTEXT, 0, (LPARAM)L"0");
		CheckDlgButton(hWnd, IDC_SHOW_BONES_CHECK, BST_CHECKED);
		hBorderSegment = GetDlgItem(hWnd, IDC_BORDER_SEGMENT_EDIT);
		SendMessage(hBorderSegment, WM_SETTEXT, 0, (LPARAM)L"-1");
		hBindBonesButton = GetDlgItem(hWnd, IDC_BIND_BUTTON);
		hAddBonesButton = GetDlgItem(hWnd, IDC_ADD_BUTTON);
		hDeleteBonesButton = GetDlgItem(hWnd, IDC_DELETE_BUTTON);
		hCopyBonesButton = GetDlgItem(hWnd, IDC_COPY_BUTTON);
		hBonesThickness = GetDlgItem(hWnd, IDC_THICKNESS_EDIT);
		SendMessage(hBonesThickness, WM_SETTEXT, 0, (LPARAM)L"10");
	}
	break;
	case WM_COMMAND:
	{
		int wmId = LOWORD(wParam);
		// Разобрать выбор в меню:
		switch (wmId)
		{
		case IDC_ADD_BUTTON:
			{
				generalModel->AddBone();
			}
			break;
		case IDC_LENGTH_BUTTON:
			{
				GetWindowTextW(hBoneLength, text, MAX_LOADSTRING);
				generalModel->SetBoneLength(wcstof(text, NULL));
			}
			break;
		case IDC_SET_THICKNESS_BUTTON:
			{
				GetWindowTextW(hBonesThickness, text, MAX_LOADSTRING);
				generalModel->SetBonesThickness(wcstof(text, NULL));
			}
			break;
		case IDC_TRANSLATE_BUTTON:
			{
				Vector3 v;
				GetWindowTextW(hXTranslate, text, MAX_LOADSTRING);
				v.x = wcstof(text, NULL);
				GetWindowTextW(hYTranslate, text, MAX_LOADSTRING);
				v.y = wcstof(text, NULL);
				GetWindowTextW(hZTranslate, text, MAX_LOADSTRING);
				v.z = wcstof(text, NULL);
				generalModel->TranslateSkeleton(v);
			}
			break;
		case IDC_ROTATE_BUTTON:
		{
			Vector3 v;
			GetWindowTextW(hXRotate, text, MAX_LOADSTRING);
			v.x = wcstof(text, NULL);
			GetWindowTextW(hYRotate, text, MAX_LOADSTRING);
			v.y = wcstof(text, NULL);
			GetWindowTextW(hZRotate, text, MAX_LOADSTRING);
			v.z = wcstof(text, NULL);
			generalModel->RotateBone(v);
		}
		break;
		case IDC_DELETE_BUTTON:
		{
			generalModel->DeleteBone();
		}
		break;
		case IDC_COPY_BUTTON:
		{
			generalModel->CopyBones();
		}
		break;
		case IDC_X_MIRROR_BUTTON:
		{
			generalModel->MirrorRotationX();
		}
		break;
		case IDC_Y_MIRROR_BUTTON:
		{
			generalModel->MirrorRotationY();
		}
		break;
		case IDC_Z_MIRROR_BUTTON:
		{
			generalModel->MirrorRotationZ();
		}
		break;
		case IDC_SHOW_BONES_CHECK:
		{
			if (SendDlgItemMessage(hWnd, IDC_SHOW_BONES_CHECK, BM_GETCHECK, 0, 0))
			{
				generalModel->SetBonesVisiblity(true);
			}
			else
			{
				generalModel->SetBonesVisiblity(false);
			}
		}
		break;
		case IDC_BIND_BUTTON:
		{
			if (generalModel->IsBonesBinded())
			{
				generalModel->UnbindBones();
				SendMessage(hBindBonesButton, WM_SETTEXT, 0, (LPARAM)L"Bind");
				EnableWindow(hAddBonesButton, TRUE);
				EnableWindow(hDeleteBonesButton, TRUE);
				EnableWindow(hCopyBonesButton, TRUE);
			}
			else
			{
				GetWindowTextW(hBorderSegment, text, MAX_LOADSTRING);
				int borderSegment = wcstof(text, NULL);
				generalModel->BindBones(borderSegment);

				SendMessage(hBindBonesButton, WM_SETTEXT, 0, (LPARAM)L"Unbind");
				EnableWindow(hAddBonesButton, FALSE);
				EnableWindow(hDeleteBonesButton, FALSE);
				EnableWindow(hCopyBonesButton, FALSE);
			}	
		}
		break;
		}
	}
	break;
	case WM_CLOSE:
		EndDialog(hWnd, 0);
		PostQuitMessage(0);
		return 0;
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

			if (isLoaded)
			{
				T = clock();

				inputController->Update();
				generalModel->Render();

				T = (clock() - T);
				totalTime += T;
				framesCount++;

				if (T != 0)
				{
					SetWindowTextA(hFPS, (to_string(CLOCKS_PER_SEC / T)).c_str());
					SetWindowTextA(hMeanFPS, (to_string((float)framesCount * CLOCKS_PER_SEC / totalTime)).c_str());
				}
			}
		}
	}
}
