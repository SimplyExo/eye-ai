package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.llm.LLM

data class CurrentSettingsSnapshot(
	val speechRate: Float,
	val voice: Int,
	val frequency: Int,
	val bps: Int
)

sealed class SettingsExtractionResult {
	data class Complete(
		val json: String,
		val confirmationQuestion: String
	) : SettingsExtractionResult()

	data class MissingValue(val targetedQuestion: String) : SettingsExtractionResult()
	data object Failed : SettingsExtractionResult()
}

/** Performs exactly one Gemini parameter-extraction request per invocation. */
class GeminiSettingsExtractor(
	private val jsonParser: JsonParser,
	private val trace: (String) -> Unit = {},
	private val generateLlmResponse: suspend (String, Boolean) -> String?
) {
	suspend fun extract(
		settingIntent: SettingIntent,
		input: String,
		currentJson: String?,
		settings: CurrentSettingsSnapshot
	): SettingsExtractionResult {
		trace(
			"[DecisionTrace][Gemini API][EVALUATE] role=SETTINGS_PARAMETER_EXTRACTION " +
				"settingIntent=$settingIntent input='$input'"
		)
		val prompt = buildPrompt(settingIntent, input, currentJson, settings)
		val jsonResponse = generateLlmResponse(prompt, true) ?: run {
			trace(
				"[DecisionTrace][Gemini API][RESULT] role=SETTINGS_PARAMETER_EXTRACTION " +
					"settingIntent=$settingIntent outcome=FAILED"
			)
			return SettingsExtractionResult.Failed
		}

		val normalizedResponse = jsonParser.normalizedExpectedSettingChange(
			jsonResponse,
			settingIntent
		) ?: run {
			trace(
				"[DecisionTrace][Gemini API][RESULT] role=SETTINGS_PARAMETER_EXTRACTION " +
					"settingIntent=$settingIntent outcome=MISSING_VALUE"
			)
			return SettingsExtractionResult.MissingValue(targetedQuestion(settingIntent))
		}

		val contextualJsonResponse = jsonParser.carrySettingsContext(normalizedResponse, currentJson)
		trace(
			"[DecisionTrace][Gemini API][RESULT] role=SETTINGS_PARAMETER_EXTRACTION " +
				"settingIntent=$settingIntent outcome=COMPLETE"
		)
		return SettingsExtractionResult.Complete(
			json = contextualJsonResponse,
			confirmationQuestion = jsonParser.createConfirmationQuestion(contextualJsonResponse)
		)
	}

	private fun buildPrompt(
		settingIntent: SettingIntent,
		input: String,
		currentJson: String?,
		settings: CurrentSettingsSnapshot
	): String {
		val originalText = jsonParser.parseSettingsOriginalText(currentJson)
		val utteranceContext = if (originalText != null && originalText != input) {
			"""
				Die vollständige ursprüngliche Äußerung war: '$originalText'
				Die aktuelle Ergänzung des Nutzers ist: '$input'
			""".trimIndent()
		} else {
			"Die vollständige Äußerung des Nutzers ist: '$input'"
		}

		val extractionRule = """
			Eine eindeutige relative Änderung wie erhöhen, verringern, schneller, langsamer
			oder die andere Stimme ist ebenfalls ein vollständiger Parameter und soll anhand
			des aktuellen Werts sinnvoll berechnet werden. Wenn weder ein konkreter Wert noch
			eine eindeutige Richtung bzw. gewünschte Stimme genannt ist, erfinde keinen Wert,
			setze 'settings_parameter_complete' auf false und gib 'changed_settings' als leeres
			Array zurück. Setze bei einem vollständigen Parameter 'settings_parameter_complete'
			auf true und gib genau ein Objekt mit ausschließlich dem erwarteten Einstellungsfeld
			zurück. Das Vollständigkeitsfeld muss immer gesetzt werden.
		""".trimIndent()

		return when (settingIntent) {
			SettingIntent.TTS_SPEED -> """
				Die aktuelle Sprechgeschwindigkeit ist ${settings.speechRate}.
				$utteranceContext
				Passe ausschließlich die Sprechgeschwindigkeit an und verwende dafür das Feld
				'tts_speed' im Array 'changed_settings'. "Geschwindigkeit ändern" ohne Wert oder
				Richtung ist unvollständig.
				$extractionRule
			""".trimIndent()

			SettingIntent.VOICE -> {
				val currentVoice = settings.voice.coerceIn(0, 1)
				val currentVoiceDescription = if (currentVoice == 1) "männlich" else "weiblich"
				"""
					Die aktuell gespeicherte Assistentenstimme ist $currentVoiceDescription (voice=$currentVoice).
					$utteranceContext
					Passe ausschließlich die Stimme an. Verwende voice=1 für männlich und voice=0
					für weiblich. Bei "andere Stimme" verwende den jeweils anderen Wert. Gib das
					Feld 'voice' im Array 'changed_settings' zurück. "Stimme ändern" ohne männlich,
					weiblich oder andere ist unvollständig.
					$extractionRule
				""".trimIndent()
			}

			SettingIntent.FREQUENCY -> """
				Die aktuelle Audio-Frequenz ist ${settings.frequency} Hz.
				$utteranceContext
				Passe ausschließlich die Audio-Frequenz an. Sie muss zwischen 100 und 4000 Hz
				liegen. Verwende das Feld 'frequency' im Array 'changed_settings'. "Frequenz ändern"
				ohne Wert oder Richtung ist unvollständig.
				$extractionRule
			""".trimIndent()

			SettingIntent.BPS -> """
				Die aktuelle Signalrate ist ${settings.bps} BPS (Beats per Second).
				$utteranceContext
				Passe ausschließlich die Signalrate an. Sie muss zwischen 1 und 10 BPS liegen.
				Langsamere Abstandssignale bedeuten weniger BPS, schnellere mehr BPS. Verwende
				das Feld 'bps' im Array 'changed_settings'. "Signalrate ändern" ohne Wert oder
				Richtung ist unvollständig.
				$extractionRule
			""".trimIndent()

			SettingIntent.LEAVE, SettingIntent.NONE -> "Führe die folgende Aktion aus: '$input'."
		}
	}

	private fun targetedQuestion(settingIntent: SettingIntent): String = when (settingIntent) {
		SettingIntent.TTS_SPEED -> LLM.SNIPPET_TTS_SPEED
		SettingIntent.VOICE -> LLM.SNIPPET_VOICE
		SettingIntent.FREQUENCY -> LLM.SNIPPET_FREQUENCY
		SettingIntent.BPS -> LLM.SNIPPET_BPS
		SettingIntent.LEAVE, SettingIntent.NONE -> "Welche Einstellung möchten Sie ändern?"
	}
}

enum class SettingsConfirmationResult {
	APPLIED,
	REJECTED,
	ABORTED,
	FAILED
}

/** Performs one Gemini approval request and invokes [applySettings] at most once. */
class GeminiSettingsConfirmation(
	private val jsonParser: JsonParser,
	private val trace: (String) -> Unit = {},
	private val generateLlmResponse: suspend (String, Boolean) -> String?
) {
	suspend fun confirmAndApply(
		input: String,
		currentJson: String?,
		applySettings: suspend (String) -> Boolean
	): SettingsConfirmationResult {
		trace(
			"[DecisionTrace][Gemini API][EVALUATE] role=SETTINGS_CONFIRMATION input='$input'"
		)
		val prompt = """
			Der Nutzer wurde gefragt, ob eine konkrete Einstellungsänderung ausgeführt werden soll.
			Die Antwort des Nutzers war: '$input'.
			Unterscheide genau diese Fälle:
			- Zustimmung zur Änderung: approval=1 und abort_settings_flow=false.
			- Nur Ablehnung dieser Änderung, zum Beispiel "Nein": approval=0 und abort_settings_flow=false.
			- Ausdrücklicher Abbruch des gesamten Einstellungsdialogs, zum Beispiel "Abbrechen" oder "Stopp":
			  approval=0 und abort_settings_flow=true.
		""".trimIndent()
		val jsonResponse = generateLlmResponse(prompt, true) ?: run {
			trace("[DecisionTrace][Gemini API][RESULT] role=SETTINGS_CONFIRMATION outcome=FAILED")
			return SettingsConfirmationResult.FAILED
		}

		if (jsonParser.isSettingsFlowAbort(jsonResponse)) {
			trace("[DecisionTrace][Gemini API][RESULT] role=SETTINGS_CONFIRMATION outcome=ABORTED")
			return SettingsConfirmationResult.ABORTED
		}
		if (!jsonParser.isApproved(jsonResponse)) {
			trace("[DecisionTrace][Gemini API][RESULT] role=SETTINGS_CONFIRMATION outcome=REJECTED")
			return SettingsConfirmationResult.REJECTED
		}
		val settingsJson = currentJson ?: run {
			trace(
				"[DecisionTrace][Gemini API][RESULT] role=SETTINGS_CONFIRMATION " +
					"outcome=FAILED reason=MISSING_SETTINGS_CONTEXT"
			)
			return SettingsConfirmationResult.FAILED
		}

		return if (applySettings(settingsJson)) {
			trace("[DecisionTrace][Gemini API][RESULT] role=SETTINGS_CONFIRMATION outcome=APPROVED_AND_APPLIED")
			SettingsConfirmationResult.APPLIED
		} else {
			trace("[DecisionTrace][Gemini API][RESULT] role=SETTINGS_CONFIRMATION outcome=APPROVED_BUT_APPLY_FAILED")
			SettingsConfirmationResult.FAILED
		}
	}
}
