#pragma once
#include "GeneralModel.h"

class InputController
{
public:
	InputController(shared_ptr<GeneralModel> generalModel);
	~InputController();
	void Update();
private:
	shared_ptr<GeneralModel> m_generalModel;
	Keyboard m_keyboard;
	Mouse m_mouse;
	Keyboard::KeyboardStateTracker m_keyboardTracker;
	Mouse::ButtonStateTracker m_mouseTracker;
	int x;
	int y;
	Vector3 delta;
};

