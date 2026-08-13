package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

enum class ContextSwitchConfirmationResult {
	APPROVED,
	REJECTED,
	ABORTED,
	FAILED
}

/** Gemini remains responsible for interpreting the user's yes/no response. */
class GeminiContextSwitchConfirmation(
	private val jsonParser: JsonParser,
	private val trace: (String) -> Unit = {},
	private val generateLlmResponse: suspend (String, Boolean) -> String?
) {
	suspend fun evaluate(
		input: String,
		pendingIntent: PendingExternalIntent
	): ContextSwitchConfirmationResult {
		val originalResult = pendingIntent.intentResult
		trace(
			"[DecisionTrace][Gemini API][EVALUATE] role=SETTINGS_CONTEXT_SWITCH_CONFIRMATION " +
				"pendingIntent=${originalResult.intent} input='$input'"
		)
		val prompt = """
			Der Nutzer befindet sich im Einstellungsmenü und wurde gefragt, ob er es
			verlassen möchte, um den zuvor erkannten Befehl auszuführen.
			Der zuvor erkannte Intent war ${originalResult.intent}.
			Die ursprüngliche Äußerung war: '${originalResult.originalText}'.
			Die Antwort auf die Rückfrage lautet: '$input'.
			Unterscheide genau diese Fälle:
			- Zustimmung zum Kontextwechsel: approval=1 und abort_settings_flow=false.
			- Nur Ablehnung des Kontextwechsels, zum Beispiel "Nein":
			  approval=0 und abort_settings_flow=false.
			- Ausdrücklicher Abbruch des gesamten Einstellungsdialogs, zum Beispiel
			  "Abbrechen" oder "Stopp": approval=0 und abort_settings_flow=true.
		""".trimIndent()
		val response = generateLlmResponse(prompt, true) ?: run {
			trace(
				"[DecisionTrace][Gemini API][RESULT] " +
					"role=SETTINGS_CONTEXT_SWITCH_CONFIRMATION outcome=FAILED"
			)
			return ContextSwitchConfirmationResult.FAILED
		}

		if (jsonParser.isSettingsFlowAbort(response)) {
			trace(
				"[DecisionTrace][Gemini API][RESULT] " +
					"role=SETTINGS_CONTEXT_SWITCH_CONFIRMATION outcome=ABORTED"
			)
			return ContextSwitchConfirmationResult.ABORTED
		}

		return when (jsonParser.parseApproval(response)) {
			true -> {
				trace(
					"[DecisionTrace][Gemini API][RESULT] " +
						"role=SETTINGS_CONTEXT_SWITCH_CONFIRMATION outcome=APPROVED"
				)
				ContextSwitchConfirmationResult.APPROVED
			}

			false -> {
				trace(
					"[DecisionTrace][Gemini API][RESULT] " +
						"role=SETTINGS_CONTEXT_SWITCH_CONFIRMATION outcome=REJECTED"
				)
				ContextSwitchConfirmationResult.REJECTED
			}

			null -> {
				trace(
					"[DecisionTrace][Gemini API][RESULT] " +
						"role=SETTINGS_CONTEXT_SWITCH_CONFIRMATION " +
						"outcome=FAILED reason=INVALID_APPROVAL_RESPONSE"
				)
				ContextSwitchConfirmationResult.FAILED
			}
		}
	}
}
