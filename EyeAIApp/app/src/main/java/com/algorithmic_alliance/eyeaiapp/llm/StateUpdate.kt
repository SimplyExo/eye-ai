package com.algorithmic_alliance.eyeaiapp.llm
import com.algorithmic_alliance.eyeaiapp.MainActivity.State

data class StateUpdate (
	val newState: State,
	val newJson: String?
)