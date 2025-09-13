package com.algorithmic_alliance.eyeaiapp.llm.statemachine

import com.algorithmic_alliance.eyeaiapp.MainActivity

data class StateUpdate (
	val newState: MainActivity.State,
	val newJson: String?
)