package com.algorithmic_alliance.eyeaiapp.nlp

import kotlin.math.abs

/**
 * NLP V2 output classes in the exact order used during training.
 *
 * An intent's [ordinal] is its index in [IntentResult.probabilities]. Keep this
 * order synchronized with the frozen `labels.json` artifacts.
 */
enum class Intent {
	TEXT_RECOGNITION,
	OBJECT_DETECTION,
	CHANGE_SPEECH_SPEED,
	CHANGE_SPEAKER,
	REDIRECT_TO_LLM,
	OPEN_SETTINGS,
	SET_FREQUENCY,
	SET_BPS,
	MEASURE_DISTANCE,
	ABORT;

	companion object {
		/** Stable class-index order of every NLP V2 BaselineCNN model. */
		val CLASS_ORDER: List<Intent> = entries.toList()
	}
}

/**
 * Complete result of one NLP V2 classification.
 *
 * [originalText] is the unchanged Vosk utterance. [confidence] is the
 * probability at [intent]'s class index. [probabilities] always contains all
 * ten class probabilities in [Intent.CLASS_ORDER].
 */
data class IntentResult(
	val intent: Intent,
	val confidence: Float,
	val originalText: String,
	val probabilities: FloatArray
) {
	init {
		require(probabilities.size == Intent.CLASS_ORDER.size) {
			"Expected ${Intent.CLASS_ORDER.size} intent probabilities, got ${probabilities.size}"
		}
		require(probabilities.all { it.isFinite() }) {
			"Intent probabilities must be finite"
		}
		require(probabilities.all { it in 0f..1f }) {
			"Intent probabilities must be between zero and one"
		}
		require(abs(probabilities.sum() - 1f) <= PROBABILITY_SUM_TOLERANCE) {
			"Intent probabilities must sum to one"
		}
		require(intent.ordinal == probabilities.indices.maxBy { probabilities[it] }) {
			"Intent must be the top-1 class"
		}
		require(confidence == probabilities[intent.ordinal]) {
			"Confidence must match the selected intent probability"
		}
	}

	fun probabilityFor(intent: Intent): Float = probabilities[intent.ordinal]

	companion object {
		private const val PROBABILITY_SUM_TOLERANCE = 1e-3f

		internal fun fromProbabilities(
			originalText: String,
			probabilities: FloatArray
		): IntentResult {
			require(probabilities.size == Intent.CLASS_ORDER.size) {
				"Expected ${Intent.CLASS_ORDER.size} intent probabilities, got ${probabilities.size}"
			}
			val probabilitySnapshot = probabilities.copyOf()
			val topIndex = probabilitySnapshot.indices.maxBy { probabilitySnapshot[it] }
			return IntentResult(
				intent = Intent.CLASS_ORDER[topIndex],
				confidence = probabilitySnapshot[topIndex],
				originalText = originalText,
				probabilities = probabilitySnapshot
			)
		}
	}
}
