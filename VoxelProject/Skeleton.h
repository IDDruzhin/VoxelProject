#pragma once

#include "Bone.h"

class Skeleton
{
public:
	Skeleton();
	~Skeleton();
	void Process();
	void SetMatricesForDraw(Matrix viewProj);
	void CopyMatricesForDraw(Matrix* dst);
	int GetBonesCount();
private:
	shared_ptr<Bone> m_root;
	Vector3 m_pos;
	vector<Matrix> m_finalTransforms;
	vector<Matrix> m_matricesForDraw;
};

