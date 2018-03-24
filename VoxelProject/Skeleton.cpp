#include "stdafx.h"
#include "Skeleton.h"


Skeleton::Skeleton()
{
	m_root = new Bone();
	m_root->SetLength(0.0f);
	m_finalTransforms.push_back(m_root->GetFinal());
	m_matricesForDraw.push_back(Matrix::Identity);
}


Skeleton::~Skeleton()
{
}

void Skeleton::Process()
{
	m_root->RefreshLocalWithPos(m_pos);
	m_finalTransforms[m_root->GetIndex()] = m_root->GetFinal();
	stack<Bone*> bonesSt;
	stack<Matrix> combinedSt;
	Bone* cur = m_root->GetChild();
	Matrix parentCombined = m_root->GetCombined();
	if (cur != nullptr)
	{
		bonesSt.push(cur);
		combinedSt.push(parentCombined);
	}
	while (!bonesSt.empty())
	{
		cur = bonesSt.top();
		parentCombined = combinedSt.top();
		bonesSt.pop();
		combinedSt.pop();
		cur->SetCombined(parentCombined);
		m_finalTransforms[cur->GetIndex()] = cur->GetFinal();
		if (cur->GetChild() != nullptr)
		{
			bonesSt.push(cur->GetChild());
			combinedSt.push(cur->GetCombined());
		}
		cur = cur->GetSibling();
		while (cur != nullptr)
		{
			bonesSt.push(cur);
			combinedSt.push(parentCombined);
			cur = cur->GetSibling();
		}
	}
}

void Skeleton::SetMatricesForDraw(Matrix viewProj)
{
	stack<Bone*> bonesSt;
	Bone* cur = m_root;
	if (cur != nullptr)
	{
		bonesSt.push(cur);
	}
	while (!bonesSt.empty())
	{
		cur = bonesSt.top();
		bonesSt.pop();
		m_matricesForDraw[cur->GetIndex()] = ((cur->GetMatrixForDraw())*viewProj).Transpose();
		if (cur->GetChild() != nullptr)
		{
			bonesSt.push(cur->GetChild());
		}
		cur = cur->GetSibling();
		while (cur != nullptr)
		{
			bonesSt.push(cur);
			cur = cur->GetSibling();
		}
	}
}

void Skeleton::CopyMatricesForDraw(Matrix * dst)
{
	memcpy(dst, &m_matricesForDraw[0], sizeof(Matrix)*m_matricesForDraw.size());
}

int Skeleton::GetBonesCount()
{
	return m_finalTransforms.size();
}
