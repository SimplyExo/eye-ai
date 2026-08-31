package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import com.algorithmic_alliance.eyeaiapp.nlp.IntentResult

/** Identifies whether a settings conversation was opened explicitly or directly. */
enum class SettingsFlow(val wireValue: String) {
	GUIDED("guided"),
	DIRECT("direct");

	companion object {
		fun fromWireValue(value: String): SettingsFlow =
			entries.firstOrNull { it.wireValue == value } ?: GUIDED
	}
}

/** Pure routing result used by IDLE before any settings side effect is performed. */
sealed class SettingsIntentRoute {
	data object NotSettings : SettingsIntentRoute()
	data object GuidedMenu : SettingsIntentRoute()
	data class Direct(
		val settingIntent: SettingIntent,
		val intentResult: IntentResult
	) : SettingsIntentRoute()
}

object SettingsIntentRouter {
	fun route(intentResult: IntentResult): SettingsIntentRoute = when (intentResult.intent) {
		Intent.OPEN_SETTINGS -> SettingsIntentRoute.GuidedMenu
		Intent.CHANGE_SPEECH_SPEED -> direct(SettingIntent.TTS_SPEED, intentResult)
		Intent.CHANGE_SPEAKER -> direct(SettingIntent.VOICE, intentResult)
		Intent.SET_FREQUENCY -> direct(SettingIntent.FREQUENCY, intentResult)
		Intent.SET_BPS -> direct(SettingIntent.BPS, intentResult)
		else -> SettingsIntentRoute.NotSettings
	}

	private fun direct(
		settingIntent: SettingIntent,
		intentResult: IntentResult
	) = SettingsIntentRoute.Direct(settingIntent, intentResult)
}
