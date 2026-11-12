package com.algorithmic_alliance.eyeaiapp.llm.statemachine

/** Defines whether the global TTS callback may arm Vosk again. */
enum class VoskRestartPolicy {
	AUTO_RESTART_AFTER_TTS, REQUIRE_MANUAL_RESTART
}

data class StateUpdate(
	val newState: EyeAIState,
	val newJson: String?,
	val voskRestartPolicy: VoskRestartPolicy = VoskRestartPolicy.REQUIRE_MANUAL_RESTART
) {
    companion object {
        /** Use only when the spoken response requests an answer in this dialog. */
        fun awaitingReply(state: EyeAIState, context: String?) =
            StateUpdate(state, context, VoskRestartPolicy.AUTO_RESTART_AFTER_TTS)
    }
}
