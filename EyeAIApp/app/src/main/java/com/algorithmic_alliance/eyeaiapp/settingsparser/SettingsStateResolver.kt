package com.algorithmic_alliance.eyeaiapp.settingsparser

/** Current values are supplied by a future integration adapter, never by ML. */
data class CurrentSettingsState(
	val frequency: Int,
	val bps: Double,
	val speechSpeed: Double,
	val speaker: SpeakerChoice
)

sealed interface ResolvedSettingValue {
	data class Numeric(val value: Double) : ResolvedSettingValue
	data class Speaker(val value: SpeakerChoice) : ResolvedSettingValue
}

data class SettingResolution(
	val target: SettingTarget,
	val requestedValue: ResolvedSettingValue?,
	val status: SettingParseStatus,
	val diagnostic: String? = null
)

/**
 * Calculates one requested target value without confirmation, persistence, TTS,
 * or a Settings write. The existing production flow is deliberately untouched.
 */
class SettingsStateResolver(
	private val config: SettingParserConfig = SettingParserConfig(),
	private val validator: SettingCommandValidator = SettingCommandValidator(config)
) {
	fun resolve(command: SettingCommand, current: CurrentSettingsState): SettingResolution {
		val validated = validator.validate(command)
		if (validated.status != SettingParseStatus.COMPLETE) {
			return SettingResolution(
				validated.target,
				null,
				validated.status,
				validated.diagnostics.joinToString(";").ifBlank { null }
			)
		}
		if (validated.target == SettingTarget.SPEAKER) {
			if (validated.operation == SettingOperation.TOGGLE && current.speaker == SpeakerChoice.UNSPECIFIED) {
				return SettingResolution(
					validated.target,
					null,
					SettingParseStatus.NEEDS_CLARIFICATION,
					"CURRENT_SPEAKER_UNSPECIFIED"
				)
			}
			val speaker = when (validated.operation) {
				SettingOperation.TOGGLE -> if (current.speaker == SpeakerChoice.MALE) SpeakerChoice.FEMALE else SpeakerChoice.MALE
				else -> requireNotNull(validated.speaker)
			}
			return SettingResolution(validated.target, ResolvedSettingValue.Speaker(speaker), SettingParseStatus.COMPLETE)
		}
		val value = resolveNumeric(validated, current)
		return if (validator.isValidAbsolute(validated.target, value)) {
			SettingResolution(validated.target, ResolvedSettingValue.Numeric(value), SettingParseStatus.COMPLETE)
		} else {
			SettingResolution(
				validated.target,
					null,
				SettingParseStatus.INVALID_VALUE,
				"RESOLVED_VALUE_OUT_OF_RANGE"
			)
		}
	}

	private fun resolveNumeric(command: SettingCommand, current: CurrentSettingsState): Double {
		if (command.operation == SettingOperation.SET_ABSOLUTE) return requireNotNull(command.numericValue)
		val currentValue = when (command.target) {
			SettingTarget.FREQUENCY -> current.frequency.toDouble()
			SettingTarget.BPS -> current.bps.toDouble()
			SettingTarget.SPEECH_SPEED -> current.speechSpeed
			SettingTarget.SPEAKER -> error("Speaker has no numeric state")
		}
		val delta = command.numericValue ?: step(command.target, requireNotNull(command.magnitude))
		return currentValue + if (command.operation == SettingOperation.DECREASE) -delta else delta
	}

	private fun step(target: SettingTarget, magnitude: ChangeMagnitude): Double = when (target) {
		SettingTarget.FREQUENCY -> config.frequencySteps.getValue(magnitude).toDouble()
		SettingTarget.BPS -> config.bpsSteps.getValue(magnitude).toDouble()
		SettingTarget.SPEECH_SPEED -> config.speechSpeedSteps.getValue(magnitude)
		SettingTarget.SPEAKER -> error("Speaker has no numeric step")
	}
}
