#pragma once
class Object
{
public:
	Object();
	~Object();
	Matrix GetWorld();
	void Scale(Vector3 ds);
	void Rotate(Vector3 dr);
	void Translate(Vector3 dt);
protected:
	Vector3 m_s;
	Quaternion m_r;
	Vector3 m_t;
};

