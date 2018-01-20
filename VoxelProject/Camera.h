#pragma once
class Camera
{
public:
	Camera(int width = 800, int height = 600, float near_ = 0.1f, float far_ = 1000.0f, float distance = 1.0f, Vector3 lookPosition = Vector3(0.0f, 0.0f, 0.0f), float fovY = XM_PI / 3.0f);
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
