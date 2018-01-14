#include "stdafx.h"
#include "Camera.h"


Camera::Camera() : m_width(800), m_height(600), m_near(0.1f), m_far(1000.0f), m_distance(3.0f), m_lookPosition(0.0f, 0.0f, 0.0f)
{
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
