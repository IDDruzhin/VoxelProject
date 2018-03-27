#pragma once
class Bone
{
public:
	Bone(float parentLength = 0, int index = 0);
	~Bone();
	shared_ptr<Bone> GetChild();
	shared_ptr<Bone> GetSibling();
	void SetCombined(Matrix parentCombined);
	void InsertChild(int index);
	void SetSibling(shared_ptr<Bone> sibling);
	void SetLength(float length);
	void SetTranslation(float t);
	Matrix GetFinal();
	Matrix GetCombined();
	Matrix GetMatrixForDraw();
	void RefreshLocal();
	void RefreshLocalWithPos(Vector3 pos);
	int GetIndex();
	void Process(Matrix parentCombined, vector<Matrix>& finalTransforms);
	void ProcessForDraw(Matrix viewProj, vector<Matrix>& matricesForDraw);
private:
	shared_ptr<Bone> m_child;
	shared_ptr<Bone> m_sibling;
	int m_index;
	Matrix m_local;
	Matrix m_combined;
	Matrix m_offset;
	float m_length;
	float m_s;
	float m_t;
	Quaternion m_r;
};

