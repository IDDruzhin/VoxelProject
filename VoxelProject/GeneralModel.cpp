#include "stdafx.h"
#include "GeneralModel.h"


GeneralModel::GeneralModel(HWND hWnd, int width, int height)
{
	//SelectedObject = 0;
	//SelectedBoneID = 0;
	m_cameraSens = 0.08f;
	m_width = width;
	m_height = height;
	m_camera = make_shared<Camera>(m_width, m_height);
	shared_ptr<D3DSystem> d3dSyst = make_shared<D3DSystem>(hWnd, m_width, m_height);
	//m_camera->SetPerspective(m_width, m_height, XM_PI / 3.0f);
	m_background = Vector3(0.5f, 0.7f, 1.0f);
	m_voxPipeline = make_unique<VoxelPipeline>(d3dSyst);
	m_voxObj = make_shared<VoxelObject>(m_voxPipeline.get());
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

void GeneralModel::Save(ofstream * F)
{
	//voxelObjects[SelectedObject].Save(F);
}

void GeneralModel::Load(ifstream * F)
{
	/*
	if (voxelObjects.size() == 0)
	{
		voxelObjects.emplace_back();
	}
	voxelObjects[SelectedObject].Load(F);
	voxelPipeline.LoadToGPU(&d3dSyst, voxelObjects[SelectedObject].VoxData.get());
	voxelObjects[SelectedObject].SetSegmentMaskFromList("D:\\SomeData\\VoxelData\\(VKH)Bones_indexes.txt", &d3dSyst);
	//voxelObjects[SelectedObject].SetSegmentMaskFromList("D:\\SomeData\\VoxelData\\(VKH)Non_skin.txt", &d3dSyst);

	d3dSyst.Reset();
	voxelPipeline.ComputeActivityMask(&d3dSyst, voxelObjects, SelectedObject);
	d3dSyst.Execute();

	d3dSyst.Wait();

	d3dSyst.Reset();
	voxelPipeline.ComputeBorderMask(&d3dSyst, voxelObjects, SelectedObject);
	d3dSyst.Execute();


	d3dSyst.Wait();
	SelectedBone = voxelObjects[SelectedObject].GetRootBone();
	*/
	
}

void GeneralModel::LoadAnatomicalAndSegmentedList(string Path)
{
	/*
	if (voxelObjects.size() == 0)
	{
		voxelObjects.emplace_back();
	}
	voxelObjects[SelectedObject].LoadAnatomicalAndSegmentedList(Path, &d3dSyst);
	voxelPipeline.LoadToGPU(&d3dSyst, voxelObjects[SelectedObject].VoxData.get());
	//voxelObjects[SelectedObject].SetSegmentMaskFromList("D:\\SomeData\\VoxelData\\(VKH)Bones_indexes.txt", &d3dSyst);

	d3dSyst.Reset();
	voxelPipeline.ComputeActivityMask(&d3dSyst, voxelObjects, SelectedObject);
	d3dSyst.Execute();
	d3dSyst.Wait();
	d3dSyst.Reset();
	voxelPipeline.ComputeBorderMask(&d3dSyst, voxelObjects, SelectedObject);
	d3dSyst.Execute();
	d3dSyst.Wait();
	SelectedBone = voxelObjects[SelectedObject].GetRootBone();
	*/
}


