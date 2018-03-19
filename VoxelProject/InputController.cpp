#include "stdafx.h"
#include "InputController.h"

InputController::InputController(shared_ptr<GeneralModel> generalModel)
{
	m_generalModel = generalModel;
}

InputController::~InputController()
{
}

void InputController::Update()
{
	auto kb = m_keyboard.GetState();
	m_keyboardTracker.Update(kb);
	auto m = m_mouse.GetState();
	m_mouseTracker.Update(m);
	if (m_mouseTracker.rightButton == m_mouseTracker.PRESSED)
	{
		delta.x = m.x;
		delta.y = m.y;
	}
	if (m_mouseTracker.rightButton == m_mouseTracker.HELD)
	{
		delta.x -= m.x;
		delta.y -= m.y;
		if (kb.LeftShift)
		{
			m_generalModel->MoveCamera(delta);
		}
		else if (kb.LeftControl)
		{
			m_generalModel->ZoomCamera(delta.y);
		}
		else
		{
			m_generalModel->RotateCamera(delta);
		}
		delta.x = m.x;
		delta.y = m.y;
	}
}
