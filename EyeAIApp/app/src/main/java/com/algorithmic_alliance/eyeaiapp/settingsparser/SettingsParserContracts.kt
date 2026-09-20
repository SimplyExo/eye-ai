package com.algorithmic_alliance.eyeaiapp.settingsparser

import com.algorithmic_alliance.eyeaiapp.nlp.Intent

/** Settings target supplied by the existing intent classifier; it is never reclassified here. */
enum class SettingTarget {
	FREQUENCY, BPS, SPEECH_SPEED, SPEAKER;

	companion object {
		fun fromIntent(intent: Intent): SettingTarget? = when (intent) {
			Intent.SET_FREQUENCY -> FREQUENCY
			Intent.SET_BPS -> BPS
			Intent.CHANGE_SPEECH_SPEED -> SPEECH_SPEED
			Intent.CHANGE_SPEAKER -> SPEAKER
			else -> null
		}
	}
}

enum class SettingOperation {
	SET_ABSOLUTE, INCREASE, DECREASE, TOGGLE, UNSPECIFIED
}

enum class SpeakerChoice {
	MALE, FEMALE, UNSPECIFIED
}

enum class ChangeMagnitude {
	SMALL, DEFAULT, LARGE
}

enum class SettingUnit {
	HZ, BPS, SPEECH_RATE
}

enum class NumberNormalizationStatus {
	SUCCESS, NO_NUMBER, AMBIGUOUS, PARTIAL_FAILURE, INVALID
}

enum class NumberOccurrenceStatus {
	SUCCESS, PARTIAL_FAILURE, INVALID
}

enum class SettingParseStatus {
	COMPLETE, NEEDS_VALUE, NEEDS_CLARIFICATION, INVALID_VALUE, INVALID_UNIT
}

data class NumberOccurrence(
	val originalText: String,
	val value: Double?,
	val start: Int? = null,
	val end: Int? = null,
	val status: NumberOccurrenceStatus,
	/** Range replaced by <NUM>; a lexical sign stays visible to the ML model. */
	val maskStart: Int? = null,
	val maskEnd: Int? = null
) {
	init {
		require(value == null || value.isFinite()) { "Numeric occurrence values must be finite" }
		require((start == null) == (end == null)) { "Occurrence offsets must both be present or absent" }
		require(start == null || (start >= 0 && requireNotNull(end) >= start)) { "Invalid occurrence offsets" }
		require((maskStart == null) == (maskEnd == null)) { "Mask offsets must both be present or absent" }
		require(maskStart == null || (maskStart >= 0 && requireNotNull(maskEnd) >= maskStart)) {
			"Invalid mask offsets"
		}
		require(
			maskStart == null || start == null || (maskStart >= start && requireNotNull(maskEnd) <= requireNotNull(
				end
			))
		) { "Mask range must be contained in occurrence range" }
		require((status == NumberOccurrenceStatus.SUCCESS) == (value != null)) {
			"Only successful occurrences expose a reliable value"
		}
	}
}

data class NumberNormalizationResult(
	val normalizedText: String,
	val values: List<Double> = emptyList(),
	val occurrences: List<NumberOccurrence> = emptyList(),
	val status: NumberNormalizationStatus = NumberNormalizationStatus.NO_NUMBER,
	val normalizerId: String,
	val normalizerVersion: String
) {
	init {
		require(values.all { it.isFinite() }) { "Numeric values must be finite" }
		require(occurrences.filter { it.status == NumberOccurrenceStatus.SUCCESS }
			.map { requireNotNull(it.value) } == values) {
			"Values must equal successful occurrences in spoken order"
		}
		require(status != NumberNormalizationStatus.NO_NUMBER || (values.isEmpty() && occurrences.isEmpty()))
		require(status != NumberNormalizationStatus.SUCCESS || values.size == 1)
		require(status != NumberNormalizationStatus.AMBIGUOUS || values.size >= 2)
		require(
			status !in setOf(
				NumberNormalizationStatus.PARTIAL_FAILURE, NumberNormalizationStatus.INVALID
			) || occurrences.any { it.status != NumberOccurrenceStatus.SUCCESS })
	}
}

interface GermanNumberNormalizer {
	fun normalize(text: String): NumberNormalizationResult
}

data class OperationPrediction(
	val operation: SettingOperation,
	val confidence: Float,
	val probabilities: FloatArray = FloatArray(SettingOperation.entries.size)
) {
	init {
		require(confidence.isFinite() && confidence in 0f..1f) {
			"Operation confidence must be in [0, 1]"
		}
		require(probabilities.size == SettingOperation.entries.size) {
			"Operation probabilities must contain ${SettingOperation.entries.size} classes"
		}
	}

	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as OperationPrediction

		if (confidence != other.confidence) return false
		if (operation != other.operation) return false
		if (!probabilities.contentEquals(other.probabilities)) return false

		return true
	}

	override fun hashCode(): Int {
		var result = confidence.hashCode()
		result = 31 * result + operation.hashCode()
		result = 31 * result + probabilities.contentHashCode()
		return result
	}
}

data class SpeakerPrediction(
	val speaker: SpeakerChoice,
	val confidence: Float,
	val probabilities: FloatArray = FloatArray(SpeakerChoice.entries.size)
) {
	init {
		require(confidence.isFinite() && confidence in 0f..1f) {
			"Speaker confidence must be in [0, 1]"
		}
		require(probabilities.size == SpeakerChoice.entries.size) {
			"Speaker probabilities must contain ${SpeakerChoice.entries.size} classes"
		}
	}

	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as SpeakerPrediction

		if (confidence != other.confidence) return false
		if (speaker != other.speaker) return false
		if (!probabilities.contentEquals(other.probabilities)) return false

		return true
	}

	override fun hashCode(): Int {
		var result = confidence.hashCode()
		result = 31 * result + speaker.hashCode()
		result = 31 * result + probabilities.contentHashCode()
		return result
	}
}

interface OperationPredictor {
	fun predictOperation(target: SettingTarget, normalizedText: String): OperationPrediction
}

interface SpeakerPredictor {
	fun predictSpeaker(target: SettingTarget, normalizedText: String): SpeakerPrediction
}

data class SettingCommand(
	val target: SettingTarget,
	val operation: SettingOperation,
	val operationConfidence: Float,
	val numericValue: Double?,
	val magnitude: ChangeMagnitude?,
	val speaker: SpeakerChoice?,
	val speakerConfidence: Float?,
	val unit: SettingUnit?,
	val status: SettingParseStatus,
	val originalText: String,
	val normalizedText: String,
	val diagnostics: List<String> = emptyList(),
	val operationProbabilities: FloatArray = FloatArray(SettingOperation.entries.size),
	val speakerProbabilities: FloatArray = FloatArray(SpeakerChoice.entries.size),
	val numberNormalizationStatus: NumberNormalizationStatus = NumberNormalizationStatus.NO_NUMBER,
	val numberOccurrences: List<NumberOccurrence> = emptyList(),
	val extractedNumericValues: List<Double> = emptyList(),
	val normalizerId: String = "unknown",
	val normalizerVersion: String = "unknown"
) {
	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as SettingCommand

		if (operationConfidence != other.operationConfidence) return false
		if (numericValue != other.numericValue) return false
		if (speakerConfidence != other.speakerConfidence) return false
		if (target != other.target) return false
		if (operation != other.operation) return false
		if (magnitude != other.magnitude) return false
		if (speaker != other.speaker) return false
		if (unit != other.unit) return false
		if (status != other.status) return false
		if (originalText != other.originalText) return false
		if (normalizedText != other.normalizedText) return false
		if (diagnostics != other.diagnostics) return false
		if (!operationProbabilities.contentEquals(other.operationProbabilities)) return false
		if (!speakerProbabilities.contentEquals(other.speakerProbabilities)) return false
		if (numberNormalizationStatus != other.numberNormalizationStatus) return false
		if (numberOccurrences != other.numberOccurrences) return false
		if (extractedNumericValues != other.extractedNumericValues) return false
		if (normalizerId != other.normalizerId) return false
		if (normalizerVersion != other.normalizerVersion) return false

		return true
	}

	override fun hashCode(): Int {
		var result = operationConfidence.hashCode()
		result = 31 * result + (numericValue?.hashCode() ?: 0)
		result = 31 * result + (speakerConfidence?.hashCode() ?: 0)
		result = 31 * result + target.hashCode()
		result = 31 * result + operation.hashCode()
		result = 31 * result + (magnitude?.hashCode() ?: 0)
		result = 31 * result + (speaker?.hashCode() ?: 0)
		result = 31 * result + (unit?.hashCode() ?: 0)
		result = 31 * result + status.hashCode()
		result = 31 * result + originalText.hashCode()
		result = 31 * result + normalizedText.hashCode()
		result = 31 * result + diagnostics.hashCode()
		result = 31 * result + operationProbabilities.contentHashCode()
		result = 31 * result + speakerProbabilities.contentHashCode()
		result = 31 * result + numberNormalizationStatus.hashCode()
		result = 31 * result + numberOccurrences.hashCode()
		result = 31 * result + extractedNumericValues.hashCode()
		result = 31 * result + normalizerId.hashCode()
		result = 31 * result + normalizerVersion.hashCode()
		return result
	}
}

/** Immutable production model dimensions and class orders. Asset hashes live in SettingsParserAssetContract. */
object SettingsTfliteContract {
	const val ARCHITECTURE = "SPECIALIZED_WORD_OPERATION_CHAR_SPEAKER"
	const val SIGNATURE_KEY = "serving_default"
	const val WORD_MAX_LEN = 32
	const val CHARACTER_MAX_LEN = 96

	const val INPUT_NAME = "token_ids"
	const val INPUT_DTYPE = "int32"
	const val OPERATION_OUTPUT_NAME = "operation"
	const val SPEAKER_OUTPUT_NAME = "speaker"
	const val NORMALIZER_ID = "text2num_rs_de"
	const val NORMALIZER_VERSION = "3.0.2"
}
