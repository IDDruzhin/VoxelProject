#include "stdafx.h"
#include "Camera.h"


Camera::Camera(int width, int height, float near_, float far_, float distance, Vector3 lookPosition) : m_width(width), m_height(height), m_near(near_), m_far(far_), m_distance(distance), m_lookPosition(lookPosition)
{
	m_projection = XMMatrixPerspectiveFovLH(m_fovY, (float)m_width / m_height, m_near, m_far);
	UpdateView();
}


Camera::~Camera()
{
}

void Camera::SetPerspective(int width, int height, float fovY)
{
	m_width = width;
	m_height = height;
	m_fovY = fovY;
	m_projection = XMMatrixPerspectiveFovLH(m_fovY, (float)m_width / m_height, m_near, m_far);
}


Matrix Camera::GetView()
{
	return m_view;
}

Matrix Camera::GetProjection()
{
	return m_projection;
}

void Camera::UpdateView()
{
	m_view = XMMatrixTranslationFromVector(-m_lookPosition)*XMMatrixRotationQuaternion(m_rotation)*XMMatrixTranslationFromVector(Vector3(0.0f, 0.0f, m_distance));
}


void Camera::Rotate(Vector3 dr)
{
	m_rotation = m_rotation * Quaternion::CreateFromYawPitchRoll(dr.x, dr.y, dr.z);
}

void Camera::Zoom(float dx)
{
	if (m_distance + dx > 0)
	{
		m_distance += dx;
	}
}
