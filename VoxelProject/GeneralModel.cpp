#include "stdafx.h"
#include "GeneralModel.h"


GeneralModel::GeneralModel(HWND hWnd, int width, int height) : m_cameraSens(0.08f), m_width(width), m_height(height), m_background(Vector3(0.5f, 0.7f, 1.0f)), m_selectedBone(0), m_pickEps(0.05f)
{
	m_camera = make_shared<Camera>(m_width, m_height);
	shared_ptr<D3DSystem> d3dSyst = make_shared<D3DSystem>(hWnd, m_width, m_height);
	m_voxPipeline = make_unique<VoxelPipeline>(d3dSyst);
}


GeneralModel::~GeneralModel()
{
}

void GeneralModel::Render()
{
	m_voxPipeline->RenderObject(m_voxObj.get(),m_camera.get(),m_selectedBone);
}

void GeneralModel::RotateCamera(Vector3 dr)
{
	m_camera->Rotate(dr * m_cameraSens);
	m_camera->UpdateView();
}

void GeneralModel::ZoomCamera(float dx)
{
	m_camera->Zoom(dx * m_cameraSens * 0.5f);
	m_camera->UpdateView();
}

void GeneralModel::MoveCamera(Vector3 dt)
{
	m_camera->Move(dt * m_cameraSens * 0.3f);
	m_camera->UpdateView();
}

void GeneralModel::LoadBin(string path)
{
	m_voxObj = make_shared<VoxelObject>(path, VoxelObject::LOADING_MODE::LOADING_MODE_BIN, m_voxPipeline.get());
	m_selectedBone = 0;
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
	m_selectedBone = 0;
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

void GeneralModel::SetClientSize(Vector2 clientSize)
{
	m_clientSize = clientSize;
}

void GeneralModel::PickBone(float x, float y)
{
	if (m_voxObj != nullptr)
	{
		m_selectedBone = m_voxObj->PickBone((x / m_clientSize.x - 0.5f)*2.0f, -(y / m_clientSize.y - 0.5f)*2.0f, m_pickEps, (m_camera->GetView() * m_camera->GetProjection()));
	}
}

void GeneralModel::AddBone()
{
	m_selectedBone = m_voxObj->AddBone(m_selectedBone);
}

void GeneralModel::SetBoneLength(float length)
{
	m_voxObj->SetBoneLength(m_selectedBone, length);
}
