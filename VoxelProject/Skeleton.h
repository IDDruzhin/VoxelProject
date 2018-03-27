#pragma once

#define MAX_BONES 256

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
	shared_ptr<Bone> FindPrev(int index);
	int PickBone(float x, float y, float eps, Matrix viewProj);
	int AddBone(int selectedIndex);
	void SetBoneLength(int selectedIndex, float length);
	void Translate(Vector3 dt);
	void RotateBone(Vector3 dr, int index);
	void DeleteBone(int index);
	void CalculateIndices();
	void InsertMirroredBones(int index, Vector3 axis);
private:
	shared_ptr<Bone> m_root;
	Vector3 m_pos;
	int m_bonesCount;
};

