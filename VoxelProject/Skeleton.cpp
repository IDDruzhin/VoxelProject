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
	shared_ptr<Bone> cur = m_root->GetChild();
	if (cur != nullptr)
	{
		cur->Process(m_root->GetCombined());
	}
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

shared_ptr<Bone> Skeleton::FindPrev(int index)
{
	return m_root->FindPrev(m_root, index);
}

int Skeleton::PickBone(float x, float y, float eps, Matrix viewProj)
{
	Vector4 p0(0.0f, 0.0f, 0.0f, 1.0f);
	Vector4 p1(1.0f, 0.0f, 0.0f, 1.0f);
	Vector4 rP0;
	Vector4 rP1;
	float dist;
	float t;
	Vector2 pos(x, y);
	Vector2 vect;
	vector<float> candidatesDist;
	vector<int> candidatesID;
	vector<Matrix> matricesForDraw;
	matricesForDraw.resize(m_bonesCount);
	SetMatricesForDraw(viewProj, &matricesForDraw[0]);
	for (int i = 0; i < m_bonesCount; i++)
	{
		rP0 = Vector4::Transform(p0, matricesForDraw[i].Transpose());
		rP1 = Vector4::Transform(p1, matricesForDraw[i].Transpose());
		rP0 /= rP0.w;
		rP1 /= rP1.w;
		vect = Vector2(rP1 - rP0);
		pos = Vector2(x, y) - Vector2(rP0);
		t = pos.Dot(vect) / vect.LengthSquared();
		if (t<0.0f || t>1.0f)
		{
			continue;
		}
		dist = Vector2::Distance(Vector2(rP0) + t * vect, Vector2(x, y));
		if (dist <= eps)
		{
			candidatesDist.push_back(dist);
			candidatesID.push_back(i);
		}
	}
	if (candidatesDist.size() > 0)
	{
		vector<float>::iterator argmin = min_element(candidatesDist.begin(), candidatesDist.end());
		int minID = distance(candidatesDist.begin(), argmin);
		return candidatesID[minID];
	}
	return 0;
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

void Skeleton::Translate(Vector3 dt)
{
	m_pos += dt;
	Process();
}

void Skeleton::RotateBone(Vector3 dr, int index)
{
	shared_ptr<Bone> cur = Find(index);
	cur->Rotate(dr);
	Process();
}

void Skeleton::DeleteBone(int index)
{
	if (index == m_root->GetIndex())
	{
		m_root->SetChild(nullptr);
	}
	else
	{
		shared_ptr<Bone> cur = Find(index);
		shared_ptr<Bone> prev = FindPrev(index);
		if (cur == prev->GetChild())
		{
			prev->SetChild(cur->GetSibling());
		}
		else
		{
			prev->SetSibling(cur->GetSibling());
		}
	}	
	CalculateIndices();
	Process();
}

void Skeleton::CalculateIndices()
{
	m_bonesCount = 0;
	m_root->CalculateIndex(m_bonesCount);
}

