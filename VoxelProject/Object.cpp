#include "stdafx.h"
#include "Object.h"


Object::Object() : m_s(1.0f, 1.0f, 1.0f)
{
}

Object::~Object()
{
}

Matrix Object::GetWorld()
{
	Matrix World = XMMatrixAffineTransformation(m_s, Vector3(0.0f, 0.0f, 0.0f), m_r, m_t);
	return World;
}

void Object::Scale(Vector3 ds)
{
	m_s *= ds;
}

void Object::Rotate(Vector3 dr)
{
	//Quaternion tmp = Quaternion::CreateFromYawPitchRoll(dr.x, dr.y, dr.z);
	Quaternion tmp = Quaternion::CreateFromYawPitchRoll(dr.y, dr.x, dr.z);
	m_r = tmp * m_r;
}

void Object::Translate(Vector3 dt)
{
	m_t += dt;
}
