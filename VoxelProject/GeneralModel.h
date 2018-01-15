#pragma once
#include "D3DSystem.h"
#include "Camera.h"

class GeneralModel
{
public:
	GeneralModel();
	~GeneralModel();
	bool Init(HWND hWnd, int _width, int _height);
	void LoadSample();
	void Update();
	void Render();
	void ComputeBones();
	void CleanUp();
	void RotateCamera(float dYaw, float dPitch);
	void RotateCamera(Vector3 dR);
	void ZoomCamera(float dx);
	void SetNextSegmentsSet();
	void AddBone();
	void InsertMirroredBone(Vector3 Axis);
	void PickBone(float x, float y);
	void SetCurBoneLength(float _length);
	void RotateCurBone(Vector3 dR);
	void TranslateCurSkeleton(Vector3 dT);
	void RemoveBone();
	void Save(ofstream* F);
	void Load(ifstream* F);
	void LoadAnatomicalAndSegmentedList(string Path);
	void ChangeCurBoneActivity();
	void SwitchRenderBones();
	void SwitchRenderVoxels();
	void SetSelectingMode(int _Mode, int4 _Rect);
	int Test();
private:
	Camera m_camera;
	D3DSystem m_d3dSyst;
	Vector3 background;
	int SelectedObject;
	int SelectedBoneID;
	Bone* SelectedBone;
	float Eps;
	int width;
	int height;
	float CameraSens;
	int SelectingMode;

};

