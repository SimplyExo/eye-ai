package com.algorithmic_alliance.eyeaiapp.settingsparser

/** Joins component results only; it has no TTS, UI, confirmation, or apply side effect. */
class SettingCommandAssembler {
	fun assemble(
		target: SettingTarget,
		originalText: String,
		numbers: NumberNormalizationResult,
		operation: OperationPrediction,
		speaker: SpeakerPrediction?,
		magnitude: MagnitudeParseResult,
		unit: UnitParseResult
	): SettingCommand {
		val diagnostics = mutableListOf<String>()
		var extractedNumericValues = numbers.values
		val numericValue = when (numbers.status) {
			NumberNormalizationStatus.SUCCESS -> normalizedNumericValueForOperation(
				numbers = numbers,
				operation = operation.operation
			).also { selected ->
				// The signed occurrence remains raw evidence. Only the command's
				// DECREASE delta is made positive under the frozen lexical-minus rule.
				if (selected != numbers.values.single()) extractedNumericValues = listOf(selected)
			}
			NumberNormalizationStatus.AMBIGUOUS -> {
				diagnostics += "AMBIGUOUS_NUMERIC_VALUES"
				null
			}
			NumberNormalizationStatus.PARTIAL_FAILURE -> {
				diagnostics += "PARTIAL_NUMBER_NORMALIZATION"
				null
			}
			NumberNormalizationStatus.INVALID -> {
				diagnostics += "INVALID_NUMBER_NORMALIZATION"
				null
			}
			NumberNormalizationStatus.NO_NUMBER -> null
		}
		if (magnitude.status == MagnitudeParseStatus.AMBIGUOUS) {
			diagnostics += magnitude.diagnostic ?: "AMBIGUOUS_MAGNITUDE"
		}
		val relevantSpeaker = if (target == SettingTarget.SPEAKER) speaker else null
		val relevantMagnitude = if (
			operation.operation in setOf(SettingOperation.INCREASE, SettingOperation.DECREASE) &&
				numericValue == null && magnitude.status == MagnitudeParseStatus.CLEAR
		) magnitude.value else null
		val status = initialStatus(target, operation.operation, numericValue, relevantSpeaker, unit, diagnostics)
		unit.diagnostic?.let(diagnostics::add)
		return SettingCommand(
			target = target,
			operation = operation.operation,
			operationConfidence = operation.confidence,
			numericValue = numericValue,
			magnitude = relevantMagnitude,
			speaker = relevantSpeaker?.speaker,
			speakerConfidence = relevantSpeaker?.confidence,
			unit = unit.unit,
			status = status,
			originalText = originalText,
			normalizedText = numbers.normalizedText,
			diagnostics = diagnostics,
			operationProbabilities = operation.probabilities.copyOf(),
			speakerProbabilities = relevantSpeaker?.probabilities?.copyOf()
				?: FloatArray(SpeakerChoice.entries.size),
			numberNormalizationStatus = numbers.status,
			numberOccurrences = numbers.occurrences,
			extractedNumericValues = extractedNumericValues,
			normalizerId = numbers.normalizerId,
			normalizerVersion = numbers.normalizerVersion
		)
	}

	private fun normalizedNumericValueForOperation(
		numbers: NumberNormalizationResult,
		operation: SettingOperation
	): Double {
		val raw = numbers.values.single()
		val occurrence = numbers.occurrences.singleOrNull {
			it.status == NumberOccurrenceStatus.SUCCESS
		}
		return if (
			operation == SettingOperation.DECREASE &&
			raw < 0.0 &&
			occurrence != null &&
			lexicalMinusPrefix.containsMatchIn(occurrence.originalText)
		) {
			-raw
		} else {
			raw
		}
	}

	private fun initialStatus(
		target: SettingTarget,
		operation: SettingOperation,
		numericValue: Double?,
		speaker: SpeakerPrediction?,
		unit: UnitParseResult,
		diagnostics: List<String>
	): SettingParseStatus = when {
		!unit.isValid -> SettingParseStatus.INVALID_UNIT
		diagnostics.isNotEmpty() -> SettingParseStatus.NEEDS_CLARIFICATION
		operation == SettingOperation.UNSPECIFIED -> SettingParseStatus.NEEDS_CLARIFICATION
		target == SettingTarget.SPEAKER && operation == SettingOperation.SET_ABSOLUTE &&
			(speaker == null || speaker.speaker == SpeakerChoice.UNSPECIFIED) -> SettingParseStatus.NEEDS_VALUE
		target == SettingTarget.SPEAKER && operation in setOf(SettingOperation.SET_ABSOLUTE, SettingOperation.TOGGLE) ->
			SettingParseStatus.COMPLETE
		target == SettingTarget.SPEAKER -> SettingParseStatus.NEEDS_CLARIFICATION
		operation == SettingOperation.SET_ABSOLUTE && numericValue == null -> SettingParseStatus.NEEDS_VALUE
		operation in setOf(SettingOperation.SET_ABSOLUTE, SettingOperation.INCREASE, SettingOperation.DECREASE) ->
			SettingParseStatus.COMPLETE
		else -> SettingParseStatus.NEEDS_CLARIFICATION
	}

	private companion object {
		val lexicalMinusPrefix = Regex("^\\s*minus(?:\\s|$)", RegexOption.IGNORE_CASE)
	}
}
