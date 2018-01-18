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
		m_generalModel->RotateCamera(delta);
		delta.x = m.x;
		delta.y = m.y;
	}
	if (m_keyboardTracker.pressed.Up)
	{
		m_generalModel->ZoomCamera(0.1f);
	}
	if (m_keyboardTracker.pressed.Down)
	{
		m_generalModel->ZoomCamera(-0.1f);
	}
}
