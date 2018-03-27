#include "stdafx.h"
#include "Skeleton.h"


Skeleton::Skeleton() : m_bonesCount(1)
{
	m_root = make_shared<Bone>();
	m_root->SetLength(0.0f);
}


Skeleton::~Skeleton()
{
}

void Skeleton::Process()
{
	m_root->RefreshLocalWithPos(m_pos);
	m_root->Process(Matrix::Identity);
}

void Skeleton::SetMatricesForDraw(Matrix viewProj, Matrix* matricesForDraw)
{
	m_root->ProcessForDraw(viewProj, matricesForDraw);
}

int Skeleton::GetBonesCount()
{
	return m_bonesCount;
}

shared_ptr<Bone> Skeleton::Find(int index)
{
	if (index == m_root->GetIndex())
	{
		return m_root;
	}
	return m_root->Find(index);
}

int Skeleton::AddBone(int selectedIndex)
{
	int curIndex = m_bonesCount;
	shared_ptr<Bone> cur = Find(selectedIndex);
	cur->InsertChild(curIndex);
	m_bonesCount++;
	Process();
	return curIndex;
}

void Skeleton::SetBoneLength(int selectedIndex, float length)
{
	shared_ptr<Bone> cur = Find(selectedIndex);
	cur->SetLength(length);
	Process();
}

