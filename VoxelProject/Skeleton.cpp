#include "stdafx.h"
#include "Skeleton.h"


Skeleton::Skeleton() : m_bonesCount(1), m_boneThickness(10.0f)
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

void Skeleton::SetMatricesForDraw(Matrix worldViewProj, Matrix* matricesForDraw)
{
	m_root->ProcessForDraw(worldViewProj, matricesForDraw, m_boneThickness);
}

void Skeleton::SetFinalMatrices(Matrix * finalMatrices)
{
	m_root->ProcessFinal(finalMatrices);
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

int Skeleton::PickBone(float x, float y, float eps, Matrix worldViewProj)
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
	SetMatricesForDraw(worldViewProj, &matricesForDraw[0]);
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
	if (m_bonesCount < MAX_BONES)
	{
		int curIndex = m_bonesCount;
		shared_ptr<Bone> cur = Find(selectedIndex);
		cur->InsertChild(curIndex);
		m_bonesCount++;
		Process();
		return curIndex;
	}
	return 0;
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

int Skeleton::CopyBones(int index)
{
	if (index != m_root->GetIndex())
	{
		shared_ptr<Bone> cur = Find(index);
		if ((m_bonesCount + cur->GetBranchBonesCount()) < MAX_BONES)
		{
			//shared_ptr<Bone> prev = FindPrev(index);
			shared_ptr<Bone> mirrCur = make_shared<Bone>(cur);
			shared_ptr<Bone> curChild = cur->GetChild();
			if (curChild)
			{
				shared_ptr<Bone> mirrChild = make_shared<Bone>(curChild);
				mirrCur->SetChild(mirrChild);
				mirrChild->ProcessCopy(curChild);
			}		
			while (cur->GetSibling())
			{
				cur = cur->GetSibling();
			}
			cur->SetSibling(mirrCur);
			CalculateIndices();
			Process();
			return mirrCur->GetIndex();
		}	
	}
	else
	{
		return 0;
	}
}

void Skeleton::MirrorRotation(int index, Vector3 axis)
{
	shared_ptr<Bone> cur = Find(index);
	cur->MirrorRotation(axis);
	Process();
}

vector<pair<Vector3, Vector3>> Skeleton::GetBonesPoints()
{
	vector<pair<Vector3, Vector3>> bonesPoints(m_bonesCount);
	m_root->SetBonePoints(&bonesPoints[0]);
	return bonesPoints;
}

void Skeleton::SetBonesThickness(float thickness)
{
	m_boneThickness = thickness;
}

void Skeleton::SetRootPos(Vector3 pos)
{
	m_pos = pos;
	Process();
}

void Skeleton::SetOffsets()
{
	m_root->ProcessOffset();
}

void Skeleton::SaveBin(ofstream & f)
{
	if (f.is_open())
	{
		f.write((char*)(&m_pos), sizeof(Vector3));
		f.write((char*)(&m_boneThickness), sizeof(float));
		f.write((char*)(&m_bonesCount), sizeof(int));
		int childsCount = 0;
		stack<shared_ptr<Bone>> bonesSt;
		shared_ptr<Bone> cur = m_root;
		bonesSt.push(cur);
		childsCount = cur->GetChildsCount();
		f.write((char*)(&childsCount), sizeof(int));
		cur->WriteBin(f);
		while (!bonesSt.empty())
		{
			cur = bonesSt.top();
			bonesSt.pop();
			cur = cur->GetChild();
			while (cur)
			{
				bonesSt.push(cur);
				childsCount = cur->GetChildsCount();
				f.write((char*)(&childsCount), sizeof(int));
				cur->WriteBin(f);
				cur = cur->GetSibling();
			}
		}
	}
	else
	{
		throw std::exception("Can`t open file");
	}
}

void Skeleton::LoadBin(ifstream & f)
{
	f.read((char*)(&m_pos), sizeof(Vector3));
	f.read((char*)(&m_boneThickness), sizeof(float));
	f.read((char*)(&m_bonesCount), sizeof(int));
	stack<shared_ptr<Bone>> bonesSt;
	stack<int> childsCountSt;
	shared_ptr<Bone> cur = m_root;
	shared_ptr<Bone> tmp;
	int childsCount = 0;
	int tmpChildsCount;
	f.read((char*)(&childsCount), sizeof(int));
	cur->LoadBin(f);
	bonesSt.push(cur);
	childsCountSt.push(childsCount);
	while (!bonesSt.empty())
	{
		childsCount = childsCountSt.top();
		childsCountSt.pop();
		cur = bonesSt.top();
		bonesSt.pop();
		for (int i = 0; i < childsCount; i++)
		{
			f.read((char*)(&tmpChildsCount), sizeof(int));
			childsCountSt.push(tmpChildsCount);
			tmp = make_shared<Bone>();
			tmp->LoadBin(f);
			cur->InsertChild(tmp);
			bonesSt.push(tmp);
		}
	}
	Process();
}

