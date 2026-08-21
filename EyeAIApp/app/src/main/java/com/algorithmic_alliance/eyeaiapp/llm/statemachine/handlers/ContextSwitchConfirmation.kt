package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationLabel
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModel

enum class ContextSwitchConfirmationResult {
	APPROVED,
	REJECTED,
	ABORTED,
	UNKNOWN,
	FAILED
}

/** Evaluates only the explicit context-switch confirmation with the local model. */
class ContextSwitchConfirmation(
	private val confirmationModelProvider: () -> ConfirmationModel,
	private val trace: (String) -> Unit = {}
) {
	fun evaluate(
		input: String,
		pendingIntent: PendingExternalIntent
	): ContextSwitchConfirmationResult {
		val intent = pendingIntent.intentResult.intent
		if (ExplicitSettingsFlowAbort.matches(input)) {
			trace(
				"[DecisionTrace][StateMachine][SETTINGS_ABORT] " +
					"state=SETTINGS_EXTERNAL_CONFIRMATION outcome=ABORTED input='$input' " +
					"evaluator=STATE_MACHINE_CONTROL modelInvoked=false apiCalled=false"
			)
			return ContextSwitchConfirmationResult.ABORTED
		}
		val question = PendingExternalIntentPresentation.confirmationQuestion(intent)
		val pendingAction = PendingExternalIntentPresentation.pendingAction(intent)
		trace(
			"[DecisionTrace][ConfirmationModel][EVALUATE] " +
				"role=SETTINGS_CONTEXT_SWITCH_CONFIRMATION " +
				"evaluator=LOCAL_CONFIRMATION_MODEL apiCalled=false pendingIntent=$intent " +
				"question='$question' pendingAction='$pendingAction' input='$input'"
		)

		val result = try {
			confirmationModelProvider().classify(question, input, pendingAction)
		} catch (error: Exception) {
			trace(
				"[DecisionTrace][ConfirmationModel][RESULT] " +
					"role=SETTINGS_CONTEXT_SWITCH_CONFIRMATION outcome=FAILED " +
					"reason=${error::class.simpleName} evaluator=LOCAL_CONFIRMATION_MODEL " +
					"apiCalled=false"
			)
			return ContextSwitchConfirmationResult.FAILED
		}

		val outcome = when (result.label) {
			ConfirmationLabel.ACCEPT -> ContextSwitchConfirmationResult.APPROVED
			ConfirmationLabel.REJECT -> ContextSwitchConfirmationResult.REJECTED
			ConfirmationLabel.UNKNOWN -> ContextSwitchConfirmationResult.UNKNOWN
		}
		trace(
			"[DecisionTrace][ConfirmationModel][RESULT] " +
				"role=SETTINGS_CONTEXT_SWITCH_CONFIRMATION outcome=$outcome " +
				"evaluator=LOCAL_CONFIRMATION_MODEL apiCalled=false " +
				result.toDecisionTraceFields()
		)
		return outcome
	}
}
