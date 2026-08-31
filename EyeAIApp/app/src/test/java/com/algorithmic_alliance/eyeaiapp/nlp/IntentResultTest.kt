package com.algorithmic_alliance.eyeaiapp.nlp

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Test

class IntentResultTest {
	@Test
	fun classOrderMatchesFrozenTrainingLabels() {
		assertEquals(
			listOf(
				"TEXT_RECOGNITION",
				"OBJECT_DETECTION",
				"CHANGE_SPEECH_SPEED",
				"CHANGE_SPEAKER",
				"REDIRECT_TO_LLM",
				"OPEN_SETTINGS",
				"SET_FREQUENCY",
				"SET_BPS",
				"MEASURE_DISTANCE",
				"ABORT"
			),
			Intent.CLASS_ORDER.map { it.name }
		)
	}

	@Test
	fun resultPreservesOriginalTextAndAllProbabilities() {
		val originalText = "  ÖFFNE, bitte die Einstellungen!  "
		val modelOutput = floatArrayOf(
			0.01f,
			0.02f,
			0.03f,
			0.04f,
			0.05f,
			0.70f,
			0.06f,
			0.02f,
			0.03f,
			0.04f
		)

		val result = IntentResult.fromProbabilities(originalText, modelOutput)

		assertEquals(Intent.OPEN_SETTINGS, result.intent)
		assertEquals(0.70f, result.confidence)
		assertEquals(originalText, result.originalText)
		assertArrayEquals(modelOutput, result.probabilities, 0f)
		assertEquals(0.70f, result.probabilityFor(Intent.OPEN_SETTINGS))
		assertNotSame(modelOutput, result.probabilities)
	}
}
