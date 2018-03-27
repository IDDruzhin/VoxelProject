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
	void RotateCamera(Vector3 dr);
	void ZoomCamera(float dx);
	void MoveCamera(Vector3 dt);
	void LoadBin(string path);
	void SaveBin(string path);
	void LoadFromImages(string path);
	void BlocksDecomposition(int blockSize);
	void SetStepSize(float ratio);
	vector<string> GetObjectSegmentsNames();
	void SetSegmentsOpacity(vector<float> &segmentsOpacity);
	vector<float> GetSegmentsOpacity();
	void SetBlocksVisiblity(bool isVisible);
	void SetInterpolationMode(VoxelPipeline::INTERPOLATION_MODE mode);

	void AddBone();
private:
	shared_ptr<Camera> m_camera;
	Vector3 m_background;
	float m_eps;
	int m_width;
	int m_height;
	float m_cameraSens;
	shared_ptr<VoxelObject> m_voxObj;
	unique_ptr<VoxelPipeline> m_voxPipeline;
	int m_selectedBone;
};

