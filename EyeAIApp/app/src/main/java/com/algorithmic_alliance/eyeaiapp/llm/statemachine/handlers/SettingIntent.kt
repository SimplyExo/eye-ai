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

/**
 * The target is already known at this point. Keep the follow-up focused on the
 * missing operation instead of repeating the general settings explanation.
 */
internal fun SettingIntent.missingOperationQuestion(): String = when (this) {
	SettingIntent.TTS_SPEED ->
		"Verstanden, Sie möchten die Sprechgeschwindigkeit anpassen. " +
			"Möchten Sie die Sprechgeschwindigkeit erhöhen, verringern oder einen konkreten Wert einstellen?"
	SettingIntent.VOICE ->
		"Verstanden, Sie möchten die Assistentenstimme anpassen. " +
			"Möchten Sie die männliche, weibliche oder die andere verfügbare Stimme verwenden?"
	SettingIntent.FREQUENCY ->
		"Verstanden, Sie möchten die Frequenz der Distanzhinweistöne anpassen. " +
			"Möchten Sie die Frequenz erhöhen, verringern oder einen konkreten Wert einstellen?"
	SettingIntent.BPS ->
		"Verstanden, Sie möchten die Schläge pro Sekunde der Distanzhinweistöne anpassen. " +
			"Möchten Sie die BPS erhöhen, verringern oder einen konkreten Wert einstellen?"
	SettingIntent.LEAVE, SettingIntent.NONE ->
		"Welche Einstellung möchten Sie ändern?"
}

internal fun SettingIntent.missingValueQuestion(): String = when (this) {
	SettingIntent.TTS_SPEED -> "Welchen konkreten Wert soll die Sprechgeschwindigkeit haben?"
	SettingIntent.VOICE -> "Soll die Assistentenstimme männlich oder weiblich sein?"
	SettingIntent.FREQUENCY -> "Welchen konkreten Frequenzwert in Hertz soll ich einstellen?"
	SettingIntent.BPS -> "Welchen konkreten BPS-Wert soll ich einstellen?"
	SettingIntent.LEAVE, SettingIntent.NONE -> "Welche Einstellung möchten Sie ändern?"
}
