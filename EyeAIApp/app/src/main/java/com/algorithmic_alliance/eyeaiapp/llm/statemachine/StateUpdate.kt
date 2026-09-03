package com.algorithmic_alliance.eyeaiapp.llm.statemachine

/** Defines whether the global TTS callback may arm Vosk again. */
enum class VoskRestartPolicy {
	AUTO_RESTART_AFTER_TTS,
	REQUIRE_MANUAL_RESTART
}

data class StateUpdate(
	val newState: EyeAIState,
	val newJson: String?,
	val voskRestartPolicy: VoskRestartPolicy = VoskRestartPolicy.AUTO_RESTART_AFTER_TTS
)
