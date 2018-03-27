#include "stdafx.h"
#include "Bone.h"


Bone::Bone(float parentLength, int index) : m_child(nullptr), m_sibling(nullptr), m_local(Matrix::Identity), m_combined(Matrix::Identity), m_offset(Matrix::Identity),
m_length(0.1f), m_s(0.015f), m_t(parentLength), m_index(index)
{
}


Bone::~Bone()
{
}

shared_ptr<Bone> Bone::GetChild()
{
	return m_child;
}

shared_ptr<Bone> Bone::GetSibling()
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
		m_child = make_shared<Bone>(m_length, index);
	}
	else
	{
		shared_ptr<Bone> cur = m_child;
		while (cur->GetSibling() != nullptr)
		{
			cur = cur->GetSibling();
		}
		cur->SetSibling(make_shared<Bone>(m_length, index));
	}
}

void Bone::SetSibling(shared_ptr<Bone> sibling)
{
	m_sibling = move(sibling);
}

void Bone::SetLength(float length)
{
	if (length >= 0)
	{
		m_length = length;
		shared_ptr<Bone> cur = m_child;
		while (cur != nullptr)
		{
			cur->SetTranslation(length);
			cur = cur->GetSibling();
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
	return (tmp * m_combined);
}

void Bone::RefreshLocal()
{
	m_local = XMMatrixAffineTransformation(Vector3(1.0f, 1.0f, 1.0f), Vector3(0.0f, 0.0f, 0.0f), m_r, Vector3(m_t, 0.0f, 0.0f));
}

void Bone::RefreshLocalWithPos(Vector3 pos)
{
	m_local = XMMatrixAffineTransformation(Vector3(1.0f, 1.0f, 1.0f), Vector3(0.0f, 0.0f, 0.0f), m_r, pos);
}

int Bone::GetIndex()
{
	return m_index;
}

void Bone::Process(Matrix parentCombined)
{
	SetCombined(parentCombined);
	if (m_sibling)
	{
		m_sibling->Process(parentCombined);
	}
	if (m_child)
	{
		m_child->Process(m_combined);
	}
}

void Bone::ProcessForDraw(Matrix viewProj, Matrix* matricesForDraw)
{
	matricesForDraw[m_index] = (GetMatrixForDraw()*viewProj).Transpose();
	if (m_sibling)
	{
		m_sibling->ProcessForDraw(viewProj, matricesForDraw);
	}
	if (m_child)
	{
		m_child->ProcessForDraw(viewProj, matricesForDraw);
	}
}

shared_ptr<Bone> Bone::Find(int index)
{
	if (m_child)
	{
		if (m_child->GetIndex() == index)
		{
			return m_child;
		}
		else
		{
			shared_ptr<Bone> cur = m_child->Find(index);
			if (cur)
			{
				return cur;
			}
			
		}
	}
	if (m_sibling)
	{
		if (m_sibling->GetIndex() == index)
		{
			return m_sibling;
		}
		else
		{	
			shared_ptr<Bone> cur = m_sibling->Find(index);
			if (cur)
			{
				return cur;
			}
		}
	}
	return nullptr;
}
