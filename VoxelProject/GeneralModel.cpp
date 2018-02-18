#include "stdafx.h"
#include "GeneralModel.h"


GeneralModel::GeneralModel(HWND hWnd, int width, int height)
{
	//SelectedObject = 0;
	//SelectedBoneID = 0;
	//int vSize = sizeof(Voxel);
	m_cameraSens = 0.08f;
	m_width = width;
	m_height = height;
	m_camera = make_shared<Camera>(m_width, m_height);
	shared_ptr<D3DSystem> d3dSyst = make_shared<D3DSystem>(hWnd, m_width, m_height);
	//m_camera->SetPerspective(m_width, m_height, XM_PI / 3.0f);
	m_background = Vector3(0.5f, 0.7f, 1.0f);
	m_voxPipeline = make_unique<VoxelPipeline>(d3dSyst);
	//m_voxObj = make_shared<VoxelObject>(m_voxPipeline.get());

	/*
	string path = "D:\\SomeData\\VoxelData\\(VKH)InputFile.txt";
	m_voxObj = make_shared<VoxelObject>(path,VoxelObject::LOADING_MODE::LOADING_MODE_SLICES, m_voxPipeline.get());
	string savePath = "D:\\SomeData\\VoxelData\\SavedVoxels\\";
	string saveName = "VKH_palette";
	m_voxObj->SaveBin(savePath, saveName);
	*/
	
	/*
	string loadPath = "D:\\SomeData\\VoxelData\\SavedVoxels\\VKH_palette.bin";
	m_voxObj = make_shared<VoxelObject>(loadPath, VoxelObject::LOADING_MODE::LOADING_MODE_BIN, m_voxPipeline.get());
	//m_voxObj->BlocksDecomposition(m_voxPipeline.get(), 64);
	//m_voxObj->BlocksDecomposition(m_voxPipeline.get(), 1024);
	//m_voxObj->BlocksDecomposition(m_voxPipeline.get(), 128);
	m_voxObj->BlocksDecomposition(m_voxPipeline.get(), 256);
	//m_voxObj->BlocksDecomposition(m_voxPipeline.get(), 2000);
	m_voxPipeline->SetStepSize(m_voxObj->GetVoxelSize(),0.5f);
	*/
	
}


GeneralModel::~GeneralModel()
{
}

/*
bool GeneralModel::Init(HWND hWnd, int _width, int _height)
{
	Eps = 0.05f;
	width = _width;
	height = _height;
	if (!d3dSyst.InitD3D(hWnd, width, height))
	{
		return false;
	}
	if (!voxelPipeline.Init(&d3dSyst))
	{
		return false;
	}
	camera.SetPerspective(width, height, XM_PI / 3.0f);
	voxelPipeline.SetProjection(camera.GetProjection().Transpose());
	//background = Vector3(1.0f, 0.0f, 0.5f);
	background = Vector3(0.5f, 0.7f, 1.0f);

	SelectingMode = 0;
	return true;
}
*/


void GeneralModel::Render()
{
	m_voxPipeline->RenderObject(m_voxObj.get(),m_camera.get());
}


void GeneralModel::RotateCamera(Vector3 dR)
{
	m_camera->Rotate(dR*m_cameraSens);
	m_camera->UpdateView();
}

void GeneralModel::ZoomCamera(float dx)
{
	m_camera->Zoom(dx);
	m_camera->UpdateView();
}
/*
void GeneralModel::AddBone()
{
	SelectedBone->InsertChild(voxelObjects[SelectedObject].GetBonesCount());
	SelectedBoneID = voxelObjects[SelectedObject].GetBonesCount();
	voxelObjects[SelectedObject].AddBone();
	voxelObjects[SelectedObject].ProcessSkeleton();
	SelectedBone = voxelObjects[SelectedObject].GetBone(SelectedBoneID);
}

void GeneralModel::InsertMirroredBone(Vector3 Axis)
{
	voxelObjects[SelectedObject].InsertMirroredBone(SelectedBoneID, Axis);
	SelectedBoneID = 0;
	SelectedBone = voxelObjects[SelectedObject].GetBone(SelectedBoneID);
}

void GeneralModel::PickBone(float x, float y)
{
	if (voxelObjects.size() > 0)
	{
		int Picked = voxelObjects[SelectedObject].PickBone((x / width - 0.5f)*2.0f, -(y / height - 0.5f)*2.0f, Eps);
		SelectedBoneID = Picked;
		SelectedBone = voxelObjects[SelectedObject].GetBone(SelectedBoneID);
	}
}

void GeneralModel::SetCurBoneLength(float _length)
{
	SelectedBone->SetLength(_length);
}

void GeneralModel::RotateCurBone(Vector3 dR)
{
	SelectedBone->Rotate(dR);
}

void GeneralModel::TranslateCurSkeleton(Vector3 dT)
{
	voxelObjects[SelectedObject].TranslateSkeleton(dT);
}

void GeneralModel::RemoveBone()
{
	voxelObjects[SelectedObject].RemoveBone(SelectedBoneID);
	SelectedBoneID = 0;
	SelectedBone = voxelObjects[SelectedObject].GetBone(SelectedBoneID);
}
*/

void GeneralModel::LoadBin(string path)
{
	m_voxObj = make_shared<VoxelObject>(path, VoxelObject::LOADING_MODE::LOADING_MODE_BIN, m_voxPipeline.get());
}

void GeneralModel::SaveBin(string path)
{
	if (m_voxObj != nullptr)
	{
		m_voxObj->SaveBin(path,"");
	}
}

void GeneralModel::LoadFromImages(string path)
{
	m_voxObj = make_shared<VoxelObject>(path, VoxelObject::LOADING_MODE::LOADING_MODE_SLICES, m_voxPipeline.get());
}

void GeneralModel::BlocksDecomposition(int blockSize)
{
	if (m_voxObj != nullptr)
	{
		m_voxObj->BlocksDecomposition(m_voxPipeline.get(), blockSize, 1);
	}
}

void GeneralModel::SetStepSize(float ratio)
{
	if (m_voxObj != nullptr)
	{
		m_voxPipeline->SetStepSize(m_voxObj->GetVoxelSize(), ratio);
	}
}

vector<string> GeneralModel::GetObjectSegmentsNames()
{
	if (m_voxObj != nullptr)
	{
		return m_voxObj->GetSegmentsNames();
	}
}

void GeneralModel::SetSegmentsOpacity(vector<float> &segmentsOpacity)
{
	if (m_voxObj != nullptr)
	{
		m_voxObj->SetSegmentsOpacity(m_voxPipeline.get(), segmentsOpacity);
	}
}

vector<float> GeneralModel::GetSegmentsOpacity()
{
	if (m_voxObj != nullptr)
	{
		return m_voxObj->GetSegmentsOpacity();
	}
}

void GeneralModel::SetBlocksVisiblity(bool isVisible)
{
	m_voxPipeline->SetBlocksVisiblity(isVisible);
}

void GeneralModel::SetInterpolationMode(VoxelPipeline::INTERPOLATION_MODE mode)
{
	m_voxPipeline->SetInterpolationMode(mode);
}


