package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.settingsparser.CurrentSettingsState
import com.algorithmic_alliance.eyeaiapp.settingsparser.LocalSettingsCommandExecution
import com.algorithmic_alliance.eyeaiapp.settingsparser.LocalSettingsParser
import com.algorithmic_alliance.eyeaiapp.settingsparser.SettingCommand
import com.algorithmic_alliance.eyeaiapp.settingsparser.SettingOperation
import com.algorithmic_alliance.eyeaiapp.settingsparser.SettingParseStatus
import com.algorithmic_alliance.eyeaiapp.settingsparser.SettingTarget
import com.algorithmic_alliance.eyeaiapp.settingsparser.SettingsCommandExecutor

/**
 * Pure settings-choice dialog routing. It deliberately owns no Gemini callback:
 * every command and every follow-up is handled by the frozen local parser.
 */
internal sealed interface LocalSettingsDialogResult {
	data class Ready(
		val execution: LocalSettingsCommandExecution.Ready,
		val confirmationJson: String,
		val confirmationQuestion: String
	) : LocalSettingsDialogResult

	data class FollowUp(
		val settingIntent: SettingIntent,
		val question: String,
		val retainedContextJson: String?,
		val command: SettingCommand? = null,
		val status: SettingParseStatus? = null,
		val diagnostic: String? = null
	) : LocalSettingsDialogResult
}

/**
 * Converts one local parser/executor result into the existing confirmation or
 * retry dialog contract. A retry carries the exact previous settings context so
 * short follow-up answers are parsed again with the same target context.
 */
internal class LocalSettingsDialogFlow(
	private val jsonParser: JsonParser,
	private val commandExecutor: SettingsCommandExecutor = SettingsCommandExecutor()
) {
	fun process(
		input: String,
		currentJson: String?,
		currentState: CurrentSettingsState,
		parser: LocalSettingsParser?
	): LocalSettingsDialogResult {
		val settingIntent = currentJson?.let(jsonParser::parseSettingIntent) ?: SettingIntent.NONE
		val target = settingIntent.toLocalTarget()
			?: return followUp(
				settingIntent = settingIntent,
				question = settingIntent.missingOperationQuestion(),
				currentJson = currentJson,
				diagnostic = "MISSING_SETTING_TARGET"
			)
		val localParser = parser ?: return localRuntimeUnavailable(settingIntent, currentJson)
		val command = localParser.parse(target, input)
		return when (val execution = commandExecutor.execute(command, currentState)) {
			is LocalSettingsCommandExecution.Ready -> {
				val confirmationJson = jsonParser.carrySettingsContext(execution.settingsJson, currentJson)
				LocalSettingsDialogResult.Ready(
					execution = execution,
					confirmationJson = confirmationJson,
					confirmationQuestion = jsonParser.createConfirmationQuestion(confirmationJson)
				)
			}

			is LocalSettingsCommandExecution.NotReady -> followUp(
				settingIntent = settingIntent,
				question = localRecoveryQuestion(
					settingIntent = settingIntent,
					status = execution.resolution.status,
					command = execution.command
				),
				currentJson = currentJson,
				command = execution.command,
				status = execution.resolution.status,
				diagnostic = execution.resolution.diagnostic
			)

			is LocalSettingsCommandExecution.UnsupportedAppRepresentation -> followUp(
				settingIntent = settingIntent,
				question = "Die aktuelle Audioausgabe unterstützt für BPS nur ganze Werte. " +
					"Bitte nennen Sie eine ganze Anzahl von Schlägen pro Sekunde.",
				currentJson = currentJson,
				command = execution.command,
				diagnostic = execution.diagnostic
			)
		}
	}

	fun localRuntimeUnavailable(currentJson: String?): LocalSettingsDialogResult.FollowUp {
		val settingIntent = currentJson?.let(jsonParser::parseSettingIntent) ?: SettingIntent.NONE
		return localRuntimeUnavailable(settingIntent, currentJson)
	}

	private fun localRuntimeUnavailable(
		settingIntent: SettingIntent,
		currentJson: String?
	): LocalSettingsDialogResult.FollowUp = followUp(
		settingIntent = settingIntent,
		question = "Die lokale Verarbeitung dieser Einstellung ist gerade nicht verfügbar. " +
			"Bitte wiederholen Sie die gewünschte Änderung.",
		currentJson = currentJson,
		diagnostic = "LOCAL_RUNTIME_UNAVAILABLE"
	)

	private fun followUp(
		settingIntent: SettingIntent,
		question: String,
		currentJson: String?,
		command: SettingCommand? = null,
		status: SettingParseStatus? = null,
		diagnostic: String? = null
	): LocalSettingsDialogResult.FollowUp = LocalSettingsDialogResult.FollowUp(
		settingIntent = settingIntent,
		question = question,
		retainedContextJson = currentJson,
		command = command,
		status = status,
		diagnostic = diagnostic
	)

	private fun SettingIntent.toLocalTarget(): SettingTarget? = when (this) {
		SettingIntent.FREQUENCY -> SettingTarget.FREQUENCY
		SettingIntent.BPS -> SettingTarget.BPS
		SettingIntent.TTS_SPEED -> SettingTarget.SPEECH_SPEED
		SettingIntent.VOICE -> SettingTarget.SPEAKER
		SettingIntent.LEAVE, SettingIntent.NONE -> null
	}

	private fun localRecoveryQuestion(
		settingIntent: SettingIntent,
		status: SettingParseStatus,
		command: SettingCommand
	): String = when (status) {
		SettingParseStatus.NEEDS_VALUE ->
			if (command.operation == SettingOperation.UNSPECIFIED) {
				settingIntent.missingOperationQuestion()
			} else {
				settingIntent.missingValueQuestion()
			}
		SettingParseStatus.INVALID_UNIT ->
			"Die genannte Einheit passt nicht zu dieser Einstellung. ${settingIntent.missingValueQuestion()}"
		SettingParseStatus.INVALID_VALUE ->
			"Der genannte Wert kann nicht verwendet werden. ${settingIntent.missingValueQuestion()}"
		SettingParseStatus.NEEDS_CLARIFICATION ->
			if (command.operation == SettingOperation.UNSPECIFIED) {
				settingIntent.missingOperationQuestion()
			} else {
				"Die gewünschte Änderung ist noch nicht eindeutig. ${settingIntent.missingValueQuestion()}"
			}
		SettingParseStatus.COMPLETE -> error("A complete local command must be prepared for confirmation")
	}

}
