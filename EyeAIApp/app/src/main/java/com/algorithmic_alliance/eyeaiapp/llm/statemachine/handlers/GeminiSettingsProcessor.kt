package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationLabel
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModel
import com.algorithmic_alliance.eyeaiapp.llm.LLM
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

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
	data class InvalidResponse(val recoveryQuestion: String) : SettingsExtractionResult()
	data object Failed : SettingsExtractionResult()
}

/** Performs one Gemini extraction request and at most one structural repair request. */
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

		val assessment = assessResponse(jsonResponse, settingIntent)
		val normalizedResponse = when (assessment) {
			is ExtractionResponseAssessment.Complete -> assessment.normalizedJson
			ExtractionResponseAssessment.MissingValue -> {
				trace(
					"[DecisionTrace][Gemini API][RESULT] role=SETTINGS_PARAMETER_EXTRACTION " +
						"settingIntent=$settingIntent outcome=MISSING_VALUE " +
						"reason=USER_PARAMETER_INCOMPLETE repairAttempted=false"
				)
				return SettingsExtractionResult.MissingValue(targetedQuestion(settingIntent))
			}
			is ExtractionResponseAssessment.Invalid -> {
				trace(
					"[DecisionTrace][Gemini API][RESULT] role=SETTINGS_PARAMETER_EXTRACTION " +
						"settingIntent=$settingIntent outcome=INVALID_RESPONSE " +
						"reason=${assessment.reason} nextEvaluator=GEMINI_API_REPAIR " +
						"repairAttempt=1"
				)
				val repairResponse = generateLlmResponse(
					buildRepairPrompt(
						settingIntent = settingIntent,
						input = input,
						settings = settings,
						invalidResponse = jsonResponse,
						failureReason = assessment.reason
					),
					true
				) ?: run {
					trace(
						"[DecisionTrace][Gemini API][RESULT] role=SETTINGS_PARAMETER_EXTRACTION " +
							"settingIntent=$settingIntent outcome=FAILED phase=REPAIR " +
							"repairAttempt=1"
					)
					return SettingsExtractionResult.Failed
				}

				when (val repairedAssessment = assessResponse(repairResponse, settingIntent)) {
					is ExtractionResponseAssessment.Complete -> {
						trace(
							"[DecisionTrace][Gemini API][REPAIR_RESULT] " +
								"role=SETTINGS_PARAMETER_EXTRACTION settingIntent=$settingIntent " +
								"outcome=COMPLETE repairAttempt=1"
						)
						repairedAssessment.normalizedJson
					}
					ExtractionResponseAssessment.MissingValue -> {
						trace(
							"[DecisionTrace][Gemini API][REPAIR_RESULT] " +
								"role=SETTINGS_PARAMETER_EXTRACTION settingIntent=$settingIntent " +
								"outcome=MISSING_VALUE reason=USER_PARAMETER_INCOMPLETE " +
								"repairAttempt=1"
						)
						return SettingsExtractionResult.MissingValue(targetedQuestion(settingIntent))
					}
					is ExtractionResponseAssessment.Invalid -> {
						trace(
							"[DecisionTrace][Gemini API][REPAIR_RESULT] " +
								"role=SETTINGS_PARAMETER_EXTRACTION settingIntent=$settingIntent " +
								"outcome=INVALID_RESPONSE reason=${repairedAssessment.reason} " +
								"repairAttempt=1 furtherRepair=false"
						)
						return SettingsExtractionResult.InvalidResponse(
							recoveryQuestion(settingIntent)
						)
					}
				}
			}
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

	private sealed class ExtractionResponseAssessment {
		data class Complete(val normalizedJson: String) : ExtractionResponseAssessment()
		data object MissingValue : ExtractionResponseAssessment()
		data class Invalid(val reason: String) : ExtractionResponseAssessment()
	}

	private fun assessResponse(
		jsonResponse: String,
		settingIntent: SettingIntent
	): ExtractionResponseAssessment {
		val root = try {
			JSONObject(jsonResponse)
		} catch (_: JSONException) {
			return ExtractionResponseAssessment.Invalid("MALFORMED_JSON")
		}

		if (!root.has("settings_parameter_complete")) {
			return ExtractionResponseAssessment.Invalid("MISSING_COMPLETENESS_FLAG")
		}
		val completionValue = root.opt("settings_parameter_complete")
		if (completionValue !is Boolean) {
			return ExtractionResponseAssessment.Invalid("INVALID_COMPLETENESS_FLAG_TYPE")
		}

		if (!completionValue) {
			val changes = root.opt("changed_settings")
			return if (changes is JSONArray && changes.length() == 0) {
				ExtractionResponseAssessment.MissingValue
			} else {
				ExtractionResponseAssessment.Invalid("INCOMPLETE_WITH_NONEMPTY_OR_MISSING_CHANGES")
			}
		}

		val normalized = jsonParser.normalizedExpectedSettingChange(jsonResponse, settingIntent)
			?: return ExtractionResponseAssessment.Invalid("EXPECTED_CHANGE_MISSING_OR_INVALID")
		return ExtractionResponseAssessment.Complete(normalized)
	}

	private fun buildRepairPrompt(
		settingIntent: SettingIntent,
		input: String,
		settings: CurrentSettingsSnapshot,
		invalidResponse: String,
		failureReason: String
	): String = """
		Die vorherige strukturierte Antwort zur Einstellungsänderung war formal ungültig.
		Fehlercode: $failureReason
		Aktive Einstellung: ${settingIntent.wireValue}
		Nutzereingabe: '$input'
		Aktuelle Werte: Sprechgeschwindigkeit=${settings.speechRate}, Stimme=${settings.voice},
		Frequenz=${settings.frequency} Hz, BPS=${settings.bps}.
		Vorherige Antwort: $invalidResponse

		Bewerte die Nutzereingabe erneut. Eine eindeutige relative Richtung wie erhöhen,
		verringern, schneller oder langsamer ist ein vollständiger Parameter. Die erneute
		Nennung der Einstellung, zum Beispiel "Erhöhe die Frequenz", macht die Antwort
		nicht unvollständig. Antworte ausschließlich mit einem JSON-Objekt. Setze
		'settings_parameter_complete' immer auf true oder false und gib 'changed_settings'
		immer als Array zurück. Bei true muss das Array genau ein Objekt mit ausschließlich
		dem Feld '${settingIntent.changedSettingKey}' enthalten. Bei false muss es leer sein.
	""".trimIndent()

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
			des aktuellen Werts sinnvoll berechnet werden. Die erneute Nennung der gewählten
			Einstellung macht die Antwort nicht unvollständig: Zum Beispiel sind "Erhöhe die
			Frequenz", "Verringere die BPS" und "Sprich schneller" jeweils vollständig.
			Wenn weder ein konkreter Wert noch
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

	private fun recoveryQuestion(settingIntent: SettingIntent): String = when (settingIntent) {
		SettingIntent.TTS_SPEED ->
			"Die gewünschte Sprechgeschwindigkeit konnte nicht zuverlässig ausgewertet werden. " +
				"Bitte sagen Sie erneut, ob sie schneller oder langsamer werden soll, oder nennen Sie einen Wert."
		SettingIntent.VOICE ->
			"Die gewünschte Stimme konnte nicht zuverlässig ausgewertet werden. " +
				"Bitte sagen Sie erneut: männlich, weiblich oder andere Stimme."
		SettingIntent.FREQUENCY ->
			"Die gewünschte Frequenzänderung konnte nicht zuverlässig ausgewertet werden. " +
				"Bitte sagen Sie erneut: Frequenz erhöhen, Frequenz verringern oder zum Beispiel 700 Hertz."
		SettingIntent.BPS ->
			"Die gewünschte BPS-Änderung konnte nicht zuverlässig ausgewertet werden. " +
				"Bitte sagen Sie erneut: BPS erhöhen, BPS verringern oder nennen Sie einen Wert."
		SettingIntent.LEAVE, SettingIntent.NONE ->
			"Die gewünschte Einstellung konnte nicht zuverlässig ausgewertet werden. Bitte versuchen Sie es erneut."
	}
}

enum class SettingsConfirmationResult {
	APPLIED,
	REJECTED,
	ABORTED,
	UNKNOWN,
	FAILED
}

/** Uses the local confirmation model and invokes [applySettings] at most once. */
class LocalSettingsConfirmation(
	private val confirmationModelProvider: () -> ConfirmationModel,
	private val jsonParser: JsonParser,
	private val trace: (String) -> Unit = {}
) {
	fun evaluate(input: String, currentJson: String?): ConfirmationLabel? {
		val settingsJson = currentJson ?: run {
			trace(
				"[DecisionTrace][ConfirmationModel][RESULT] role=SETTINGS_CONFIRMATION " +
					"outcome=FAILED reason=MISSING_SETTINGS_CONTEXT " +
					"evaluator=LOCAL_CONFIRMATION_MODEL apiCalled=false"
			)
			return null
		}
		val question = jsonParser.createConfirmationQuestion(settingsJson)
		val pendingAction = jsonParser.createPendingActionDescription(settingsJson)
		trace(
			"[DecisionTrace][ConfirmationModel][EVALUATE] " +
				"role=SETTINGS_CONFIRMATION evaluator=LOCAL_CONFIRMATION_MODEL " +
				"apiCalled=false question='$question' pendingAction='$pendingAction' input='$input'"
		)
		return try {
			val result = confirmationModelProvider().classify(question, input, pendingAction)
			trace(
				"[DecisionTrace][ConfirmationModel][RESULT] role=SETTINGS_CONFIRMATION " +
					"evaluator=LOCAL_CONFIRMATION_MODEL apiCalled=false " +
					result.toDecisionTraceFields()
			)
			result.label
		} catch (error: Exception) {
			trace(
				"[DecisionTrace][ConfirmationModel][RESULT] role=SETTINGS_CONFIRMATION " +
					"outcome=FAILED reason=${error::class.simpleName} " +
					"evaluator=LOCAL_CONFIRMATION_MODEL apiCalled=false"
			)
			null
		}
	}

	suspend fun confirmAndApply(
		input: String,
		currentJson: String?,
		applySettings: suspend (String) -> Boolean
	): SettingsConfirmationResult {
		if (ExplicitSettingsFlowAbort.matches(input)) {
			trace(
				"[DecisionTrace][StateMachine][SETTINGS_ABORT] " +
					"state=SETTINGS_ACTION outcome=ABORTED input='$input' " +
					"evaluator=STATE_MACHINE_CONTROL modelInvoked=false apiCalled=false"
			)
			return SettingsConfirmationResult.ABORTED
		}
		val settingsJson = currentJson ?: run {
			return SettingsConfirmationResult.FAILED
		}
		when (evaluate(input, settingsJson)) {
			ConfirmationLabel.REJECT -> {
				trace(
					"[DecisionTrace][ConfirmationModel][FINAL_ACTION] " +
						"role=SETTINGS_CONFIRMATION decision=REJECT " +
						"sideEffect=SKIP_SETTINGS_APPLY apiCalled=false"
				)
				return SettingsConfirmationResult.REJECTED
			}
			ConfirmationLabel.UNKNOWN -> {
				trace(
					"[DecisionTrace][ConfirmationModel][FINAL_ACTION] " +
						"role=SETTINGS_CONFIRMATION decision=UNKNOWN " +
						"sideEffect=NONE clarificationRequired=true apiCalled=false"
				)
				return SettingsConfirmationResult.UNKNOWN
			}
			ConfirmationLabel.ACCEPT -> trace(
				"[DecisionTrace][ConfirmationModel][FINAL_ACTION] " +
					"role=SETTINGS_CONFIRMATION decision=ACCEPT " +
					"sideEffect=REQUEST_SETTINGS_APPLY apiCalled=false"
			)
			null -> return SettingsConfirmationResult.FAILED
		}

		return if (applySettings(settingsJson)) {
			trace(
				"[DecisionTrace][ConfirmationModel][APPLY] " +
					"role=SETTINGS_CONFIRMATION decision=ACCEPT " +
					"outcome=APPROVED_AND_APPLIED apiCalled=false"
			)
			SettingsConfirmationResult.APPLIED
		} else {
			trace(
				"[DecisionTrace][ConfirmationModel][APPLY] " +
					"role=SETTINGS_CONFIRMATION decision=ACCEPT " +
					"outcome=APPROVED_BUT_APPLY_FAILED apiCalled=false"
			)
			SettingsConfirmationResult.FAILED
		}
	}
}
