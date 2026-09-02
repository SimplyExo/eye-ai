package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationLabel
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModel

enum class SettingsConfirmationResult {
	APPLIED,
	REJECTED,
	NOT_APPLIED,
	UNKNOWN,
	FAILED
}

enum class SettingsApplyResult {
	APPLIED,
	NOT_APPLIED,
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
	): SettingsConfirmationResult = confirmAndApplyWithResult(input, currentJson) { settingsJson ->
			if (applySettings(settingsJson)) SettingsApplyResult.APPLIED else SettingsApplyResult.FAILED
		}

	suspend fun confirmAndApplyWithResult(
		input: String,
		currentJson: String?,
		applySettings: suspend (String) -> SettingsApplyResult
	): SettingsConfirmationResult {
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

		return when (applySettings(settingsJson)) {
			SettingsApplyResult.APPLIED -> {
				trace(
					"[DecisionTrace][ConfirmationModel][APPLY] " +
						"role=SETTINGS_CONFIRMATION decision=ACCEPT " +
						"outcome=APPROVED_AND_APPLIED apiCalled=false"
				)
				SettingsConfirmationResult.APPLIED
			}
			SettingsApplyResult.NOT_APPLIED -> {
				trace(
					"[DecisionTrace][ConfirmationModel][APPLY] " +
						"role=SETTINGS_CONFIRMATION decision=ACCEPT " +
						"outcome=APPROVED_BUT_NOT_APPLIED apiCalled=false"
				)
				SettingsConfirmationResult.NOT_APPLIED
			}
			SettingsApplyResult.FAILED -> {
				trace(
					"[DecisionTrace][ConfirmationModel][APPLY] " +
						"role=SETTINGS_CONFIRMATION decision=ACCEPT " +
						"outcome=APPROVED_BUT_APPLY_FAILED apiCalled=false"
				)
				SettingsConfirmationResult.FAILED
			}
		}
	}
}
