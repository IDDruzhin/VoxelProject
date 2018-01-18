#pragma once
#include "D3DSystem.h"
#include "Camera.h"
#include "VoxelObject.h"
#include "VoxelPipeline.h"

class GeneralModel
{
public:
	GeneralModel(HWND hWnd, int width, int height);
	~GeneralModel();
	void Render();
	void RotateCamera(Vector3 dR);
	void ZoomCamera(float dx);
	void Save(ofstream* F);
	void Load(ifstream* F);
	void LoadAnatomicalAndSegmentedList(string Path);
	//void AddBone();
	//void InsertMirroredBone(Vector3 Axis);
	//void PickBone(float x, float y);
	//void SetCurBoneLength(float _length);
	//void RotateCurBone(Vector3 dR);
	//void TranslateCurSkeleton(Vector3 dT);
	//void RemoveBone();
private:
	Camera m_camera;
	Vector3 m_background;
	float m_eps;
	int m_width;
	int m_height;
	float m_cameraSens;
	shared_ptr<VoxelObject> m_voxObj;
	unique_ptr<VoxelPipeline> m_voxPipeline;
	//D3DSystem m_d3dSyst;
	//int SelectedObject;
	//int SelectedBoneID;
	//Bone* SelectedBone;
};

