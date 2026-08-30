package com.algorithmic_alliance.eyeaiapp.settingsparser

import kotlin.math.abs
import kotlin.math.round

/** Central non-production bounds and relative steps for later StateMachine integration. */
data class SettingParserConfig(
	val frequencyRange: IntRange = 100..4000,
	/** Standalone BPS domain: a finite continuous rate, not an integer counter. */
	val bpsMin: Double = 1.0,
	val bpsMax: Double = 10.0,
	/** Null means no quantization; the future audio adapter may provide one. */
	val bpsPrecision: Double? = null,
	/** Provisional until supported Android TTS engines are verified in Phase E. */
	val speechSpeedMin: Double = 0.1,
	val speechSpeedMax: Double = 2.0,
	val frequencySteps: Map<ChangeMagnitude, Int> = mapOf(
		ChangeMagnitude.SMALL to 50,
		ChangeMagnitude.DEFAULT to 100,
		ChangeMagnitude.LARGE to 250
	),
	val bpsSteps: Map<ChangeMagnitude, Double> = mapOf(
		ChangeMagnitude.SMALL to 1.0,
		ChangeMagnitude.DEFAULT to 1.0,
		ChangeMagnitude.LARGE to 3.0
	),
	val speechSpeedSteps: Map<ChangeMagnitude, Double> = mapOf(
		ChangeMagnitude.SMALL to 0.1,
		ChangeMagnitude.DEFAULT to 0.2,
		ChangeMagnitude.LARGE to 0.4
	)
) {

	init {
		require(bpsMin.isFinite() && bpsMax.isFinite() && bpsMin <= bpsMax) {
			"BPS bounds must be finite and ordered"
		}
		require(bpsPrecision == null || (bpsPrecision.isFinite() && bpsPrecision > 0.0)) {
			"BPS precision must be positive and finite when configured"
		}
	}
}

class SettingCommandValidator(
	private val config: SettingParserConfig = SettingParserConfig()
) {
	fun validate(command: SettingCommand): SettingCommand {
		if (command.status == SettingParseStatus.INVALID_UNIT) return command
		val clarificationDiagnostics = setOf(
			"AMBIGUOUS_NUMERIC_VALUES",
			"PARTIAL_NUMBER_NORMALIZATION",
			"INVALID_NUMBER_NORMALIZATION",
			"NEGATED_MAGNITUDE",
			"CONFLICTING_MAGNITUDE_MARKERS",
			"AMBIGUOUS_MAGNITUDE"
		)
		if (command.diagnostics.any { it in clarificationDiagnostics }) {
			return withStatus(command, SettingParseStatus.NEEDS_CLARIFICATION)
		}
		if (command.operation !in allowedOperations(command.target)) {
			return withStatus(
				command,
				SettingParseStatus.NEEDS_CLARIFICATION,
				"OPERATION_NOT_ALLOWED_FOR_TARGET"
			)
		}
		if (command.operation == SettingOperation.UNSPECIFIED) {
			return withStatus(command, SettingParseStatus.NEEDS_CLARIFICATION)
		}
		if (command.target != SettingTarget.SPEAKER && command.speaker != null) {
			return withStatus(
				command,
				SettingParseStatus.NEEDS_CLARIFICATION,
				"SPEAKER_NOT_ALLOWED_FOR_TARGET"
			)
		}
		if (command.target == SettingTarget.SPEAKER && command.magnitude != null) {
			return withStatus(
				command,
				SettingParseStatus.NEEDS_CLARIFICATION,
				"MAGNITUDE_NOT_ALLOWED_FOR_SPEAKER"
			)
		}
		val expectedUnit = SettingUnitParser.defaultUnit(command.target)
		if (expectedUnit != null && command.unit == null) {
			return withStatus(command, SettingParseStatus.INVALID_UNIT, "UNIT_REQUIRED_FOR_NUMERIC_TARGET")
		}
		if (command.unit != null && command.unit != expectedUnit) {
			return withStatus(command, SettingParseStatus.INVALID_UNIT, "INVALID_UNIT_FOR_TARGET")
		}
		return if (command.target == SettingTarget.SPEAKER) validateSpeaker(command) else validateNumeric(command)
	}

	private fun validateSpeaker(command: SettingCommand): SettingCommand = when (command.operation) {
		SettingOperation.TOGGLE -> {
			if (command.numericValue != null) {
				withStatus(command, SettingParseStatus.INVALID_VALUE, "NUMERIC_VALUE_NOT_ALLOWED_FOR_SPEAKER")
			} else if (command.speaker != null && command.speaker != SpeakerChoice.UNSPECIFIED) {
				withStatus(command, SettingParseStatus.NEEDS_CLARIFICATION, "TOGGLE_SPEAKER_MUST_BE_UNSPECIFIED")
			} else {
				withStatus(command, SettingParseStatus.COMPLETE)
			}
		}
		SettingOperation.SET_ABSOLUTE -> when {
			command.numericValue != null -> withStatus(
				command,
				SettingParseStatus.INVALID_VALUE,
				"NUMERIC_VALUE_NOT_ALLOWED_FOR_SPEAKER"
			)
			command.speaker == null || command.speaker == SpeakerChoice.UNSPECIFIED ->
				withStatus(command, SettingParseStatus.NEEDS_VALUE, "SPEAKER_REQUIRED")
			else -> withStatus(command, SettingParseStatus.COMPLETE)
		}
		else -> withStatus(command, SettingParseStatus.NEEDS_CLARIFICATION)
	}

	private fun validateNumeric(command: SettingCommand): SettingCommand = when (command.operation) {
		SettingOperation.SET_ABSOLUTE -> when {
			command.magnitude != null -> withStatus(
				command,
				SettingParseStatus.NEEDS_CLARIFICATION,
				"MAGNITUDE_ONLY_ALLOWED_FOR_RELATIVE_OPERATION"
			)
			command.numericValue == null -> withStatus(command, SettingParseStatus.NEEDS_VALUE, "NUMERIC_VALUE_REQUIRED")
			!isValidAbsolute(command.target, command.numericValue) ->
				withStatus(command, SettingParseStatus.INVALID_VALUE, "ABSOLUTE_VALUE_OUT_OF_RANGE")
			else -> withStatus(command, SettingParseStatus.COMPLETE)
		}
		SettingOperation.INCREASE,
		SettingOperation.DECREASE -> when {
			command.numericValue != null && command.magnitude != null ->
				withStatus(command, SettingParseStatus.NEEDS_CLARIFICATION, "NUMERIC_DELTA_AND_MAGNITUDE")
			command.numericValue != null && !isValidDelta(command.target, command.numericValue) ->
				withStatus(command, SettingParseStatus.INVALID_VALUE, "INVALID_RELATIVE_DELTA")
			command.numericValue != null -> withStatus(command, SettingParseStatus.COMPLETE)
			command.magnitude in setOf(ChangeMagnitude.SMALL, ChangeMagnitude.DEFAULT, ChangeMagnitude.LARGE) ->
				withStatus(command, SettingParseStatus.COMPLETE)
			else -> withStatus(command, SettingParseStatus.NEEDS_VALUE, "DELTA_OR_MAGNITUDE_REQUIRED")
		}
		else -> withStatus(command, SettingParseStatus.NEEDS_CLARIFICATION)
	}

	fun isValidAbsolute(target: SettingTarget, value: Double): Boolean = when (target) {
		SettingTarget.FREQUENCY -> value.isWholeNumber() && value.toInt() in config.frequencyRange
		SettingTarget.BPS -> value.isFinite() && value in config.bpsMin..config.bpsMax && isBpsPrecision(value)
		SettingTarget.SPEECH_SPEED -> value in config.speechSpeedMin..config.speechSpeedMax
		SettingTarget.SPEAKER -> false
	}

	private fun isValidDelta(target: SettingTarget, value: Double): Boolean =
		value.isFinite() && value > 0.0 && when (target) {
			SettingTarget.FREQUENCY -> value.isWholeNumber()
			SettingTarget.BPS -> isBpsPrecision(value)
			SettingTarget.SPEECH_SPEED -> true
			SettingTarget.SPEAKER -> false
		}

	private fun isBpsPrecision(value: Double): Boolean {
		val precision = config.bpsPrecision ?: return true
		val quotient = value / precision
		return abs(quotient - round(quotient)) <= 1e-9
	}

	private fun allowedOperations(target: SettingTarget): Set<SettingOperation> = when (target) {
		SettingTarget.FREQUENCY,
		SettingTarget.BPS,
		SettingTarget.SPEECH_SPEED -> setOf(
			SettingOperation.SET_ABSOLUTE,
			SettingOperation.INCREASE,
			SettingOperation.DECREASE,
			SettingOperation.UNSPECIFIED
		)
		SettingTarget.SPEAKER -> setOf(
			SettingOperation.SET_ABSOLUTE,
			SettingOperation.TOGGLE,
			SettingOperation.UNSPECIFIED
		)
	}

	private fun withStatus(
		command: SettingCommand,
		status: SettingParseStatus,
		diagnostic: String? = null
	): SettingCommand = command.copy(
		status = status,
		diagnostics = if (diagnostic != null && diagnostic !in command.diagnostics) {
			command.diagnostics + diagnostic
		} else {
			command.diagnostics
		}
	)

	private fun Double.isWholeNumber(): Boolean = isFinite() && this % 1.0 == 0.0
}
