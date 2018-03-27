#pragma once

#include "Bone.h"

class Skeleton
{
public:
	Skeleton();
	~Skeleton();
	void Process();
	void SetMatricesForDraw(Matrix viewProj, Matrix* matricesForDraw);
	int GetBonesCount();
	shared_ptr<Bone> Find(int index);
	int PickBone(float x, float y, float eps, Matrix viewProj);
	int AddBone(int selectedIndex);
	void SetBoneLength(int selectedIndex, float length);
private:
	shared_ptr<Bone> m_root;
	Vector3 m_pos;
	int m_bonesCount;
};

