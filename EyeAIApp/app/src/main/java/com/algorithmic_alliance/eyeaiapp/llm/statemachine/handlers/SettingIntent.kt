package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

enum class SettingIntent(
	val wireValue: String,
	val changedSettingKey: String? = null
) {
	TTS_SPEED("tts_speed", "tts_speed"),
	VOICE("voice", "voice"),
	LEAVE("leave", "leave"),
	FREQUENCY("frequency", "frequency"),
	BPS("bps", "bps"),
	NONE("none");

	companion object {
		fun fromWireValue(value: String): SettingIntent =
			entries.firstOrNull { it.wireValue == value } ?: NONE
	}
}
