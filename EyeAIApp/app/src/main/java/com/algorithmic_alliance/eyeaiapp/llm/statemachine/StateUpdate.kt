package com.algorithmic_alliance.eyeaiapp.llm.statemachine

import com.algorithmic_alliance.eyeaiapp.MainActivity

/** Defines whether the global TTS callback may arm Vosk again. */
enum class VoskRestartPolicy {
	AUTO_RESTART_AFTER_TTS,
	REQUIRE_MANUAL_RESTART
}

data class StateUpdate(
	val newState: MainActivity.State,
	val newJson: String?,
	val voskRestartPolicy: VoskRestartPolicy = VoskRestartPolicy.AUTO_RESTART_AFTER_TTS
)
