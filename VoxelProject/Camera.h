#pragma once
class Camera
{
public:
	Camera();
	~Camera();
	void SetPerspective(int width, int height, float fovY);
	Matrix GetView();
	Matrix GetProjection();
	void UpdateView();
	void Rotate(Vector3 dr);
	void Zoom(float dx);
private:
	Matrix m_view;
	Matrix m_projection;
	int m_width;
	int m_height;
	float m_fovY;
	float m_near;
	float m_far;
	float m_distance;
	Vector3 m_lookPosition;
	Quaternion m_rotation;
};
