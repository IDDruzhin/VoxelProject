#include "stdafx.h"
#include "Bone.h"


Bone::Bone(float parentLength, int index) : m_child(nullptr), m_sibling(nullptr), m_local(Matrix::Identity), m_combined(Matrix::Identity), m_offset(Matrix::Identity),
m_length(1.0f), m_s(0.05f), m_t(parentLength), m_index(index)
{
}


Bone::~Bone()
{
}

Bone * Bone::GetChild()
{
	return m_child;
}

Bone * Bone::GetSibling()
{
	return m_sibling;
}

void Bone::SetCombined(Matrix parentCombined)
{
	RefreshLocal();
	m_combined = m_local * parentCombined;
}

void Bone::InsertChild(int index)
{
	if (m_child == nullptr)
	{
		m_child = new Bone(m_length, index);
	}
	else
	{
		Bone* cur = m_child;
		while (cur->GetSibling() != nullptr)
		{
			cur = cur->GetSibling();
		}
		cur->SetSibling(new Bone(m_length, index));
	}
}

void Bone::SetSibling(Bone * sibling)
{
	m_sibling = sibling;
}

void Bone::SetLength(float length)
{
	if (length >= 0)
	{
		m_length = length;
		if (m_sibling)
		{
			m_sibling->SetTranslation(length);
		}
		if (m_child)
		{
			m_child->SetTranslation(length);
		}
	}
}

void Bone::SetTranslation(float t)
{
	m_t = t;
}

Matrix Bone::GetFinal()
{
	return m_offset * m_combined;
}

Matrix Bone::GetCombined()
{
	return m_combined;
}

Matrix Bone::GetMatrixForDraw()
{
	Matrix tmp = Matrix::CreateScale(m_length, m_s, m_s);
	return (tmp*m_combined);
}

void Bone::RefreshLocal()
{
	m_local = XMMatrixAffineTransformation(Vector3(1.0f, 1.0f, 1.0f), Vector3(0.0f, 0.0f, 0.0f), m_r, Vector3(m_t, 0.0f, 0.0f));
}

void Bone::RefreshLocalWithPos(Vector3 pos)
{
	m_local = XMMatrixAffineTransformation(Vector3(1.0f, 1.0f, 1.0f), Vector3(0.0f, 0.0f, 0.0f), m_r, pos);
	m_combined = m_local;
}

int Bone::GetIndex()
{
	return m_index;
}
