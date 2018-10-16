#include "stdafx.h"
#include "Bone.h"


Bone::Bone(float parentLength, int index) : m_child(nullptr), m_sibling(nullptr), m_local(Matrix::Identity), m_combined(Matrix::Identity), m_offset(Matrix::Identity),
m_length(100.0f), m_t(parentLength), m_index(index)
{
}

Bone::Bone(shared_ptr<Bone> copy)
{
	m_t = copy->m_t;
	m_r = copy->m_r;
	m_length = copy->m_length;
	m_child = nullptr;
	m_sibling = nullptr;	
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

void Bone::InsertChild(shared_ptr<Bone> child)
{
	if (m_child == nullptr)
	{
		m_child = child;
	}
	else
	{
		shared_ptr<Bone> cur = m_child;
		while (cur->GetSibling() != nullptr)
		{
			cur = cur->GetSibling();
		}
		cur->SetSibling(child);
	}
}

void Bone::SetChild(shared_ptr<Bone> child)
{
	m_child = child;
}

void Bone::SetSibling(shared_ptr<Bone> sibling)
{
	m_sibling = sibling;
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

float Bone::GetLength()
{
	return m_length;
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

Matrix Bone::GetMatrixForDraw(float thickness)
{
	Matrix tmp = Matrix::CreateScale(m_length, thickness, thickness);
	return (tmp * m_combined);
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

void Bone::ProcessForDraw(Matrix worldViewProj, Matrix* matricesForDraw, float thickness)
{
	matricesForDraw[m_index] = (GetMatrixForDraw(thickness)*worldViewProj).Transpose();
	if (m_sibling)
	{
		m_sibling->ProcessForDraw(worldViewProj, matricesForDraw, thickness);
	}
	if (m_child)
	{
		m_child->ProcessForDraw(worldViewProj, matricesForDraw, thickness);
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

shared_ptr<Bone> Bone::FindPrev(shared_ptr<Bone> prev, int index)
{
	if (m_child)
	{
		if (m_child->GetIndex() == index)
		{
			return prev;
		}
		else
		{
			shared_ptr<Bone> cur = m_child->FindPrev(m_child, index);
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
			return prev;
		}
		else
		{
			shared_ptr<Bone> cur = m_sibling->FindPrev(m_sibling, index);
			if (cur)
			{
				return cur;
			}
		}
	}
	return nullptr;
}

void Bone::Rotate(Vector3 dr)
{
	m_r = Quaternion::CreateFromYawPitchRoll(dr.y, dr.x, dr.z) * m_r;
}

void Bone::CalculateIndex(int & index)
{
	m_index = index;
	index++;
	if (m_child)
	{
		m_child->CalculateIndex(index);
	}
	if (m_sibling)
	{
		m_sibling->CalculateIndex(index);
	}
}

void Bone::ProcessCopy(shared_ptr<Bone> origin)
{
	if (origin->GetChild())
	{
		shared_ptr<Bone> child = make_shared<Bone>(origin->GetChild());
		SetChild(child);
		child->ProcessCopy(origin->GetChild());
	}
	if (origin->GetSibling())
	{
		shared_ptr<Bone> sibling = make_shared<Bone>(origin->GetSibling());
		SetSibling(sibling);
		sibling->ProcessCopy(origin->GetSibling());
	}
}

void Bone::MirrorRotation(Vector3 axis)
{
	if (axis.x == 1)
	{
		m_r.x *= -1;
	}
	if (axis.y == 1)
	{
		m_r.y *= -1;
	}
	if (axis.z == 1)
	{
		m_r.z *= -1;
	}
}

void Bone::SetBonePoints(pair<Vector3, Vector3> * bonesPoints)
{
	pair<Vector3, Vector3> p;
	p.first = Vector3(0.0f, 0.0f, 0.0f);
	p.second = Vector3(m_length, 0.0f, 0.0f);
	p.first = Vector3::Transform(p.first, m_combined);
	p.second = Vector3::Transform(p.second, m_combined);
	bonesPoints[m_index] = p;
	if (m_child)
	{
		m_child->SetBonePoints(bonesPoints);
	}
	if (m_sibling)
	{
		m_sibling->SetBonePoints(bonesPoints);
	}
}

int Bone::GetChildsCount()
{
	int count = 0;
	shared_ptr<Bone> cur = m_child;
	while (cur)
	{
		count++;
		cur = cur->GetSibling();
	}
	return count;
}

int Bone::GetBranchBonesCount()
{
	int count = 1;
	shared_ptr<Bone> cur = m_child;
	while (cur)
	{
		count += cur->GetBranchBonesCount();
		cur = cur->GetSibling();
	}
	return count;
}

void Bone::ProcessOffset()
{
	m_offset = m_combined.Invert();
	if (m_sibling)
	{
		m_sibling->ProcessOffset();
	}
	if (m_child)
	{
		m_child->ProcessOffset();
	}
}

void Bone::ProcessFinal(Matrix * finalMatrices)
{
	finalMatrices[m_index] = (m_offset * m_combined).Transpose();
	if (m_sibling)
	{
		m_sibling->ProcessFinal(finalMatrices);
	}
	if (m_child)
	{
		m_child->ProcessFinal(finalMatrices);
	}
}

void Bone::WriteBin(ofstream & f)
{
	f.write((char*)(&m_index), sizeof(int));
	f.write((char*)(&m_length), sizeof(float));
	f.write((char*)(&m_t), sizeof(float));
	f.write((char*)(&m_r), sizeof(Quaternion));
	f.write((char*)(&m_offset), sizeof(Matrix));
}

void Bone::LoadBin(ifstream & f)
{
	f.read((char*)(&m_index), sizeof(int));
	f.read((char*)(&m_length), sizeof(float));
	f.read((char*)(&m_t), sizeof(float));
	f.read((char*)(&m_r), sizeof(Quaternion));
	f.read((char*)(&m_offset), sizeof(Matrix));
}
