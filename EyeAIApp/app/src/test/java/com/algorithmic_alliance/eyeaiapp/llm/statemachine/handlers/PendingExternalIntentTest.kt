package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import com.algorithmic_alliance.eyeaiapp.nlp.IntentResult
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PendingExternalIntentTest {
	@Test
	fun codecPreservesCompleteOriginalIntentResultAcrossStateMachineInstances() {
		val originalText = "  Lies bitte das \"Schild\".\n"
		val probabilities = floatArrayOf(
			0.64f, 0.05f, 0.04f, 0.03f, 0.06f,
			0.02f, 0.05f, 0.03f, 0.04f, 0.04f
		)
		val original = IntentResult(
			intent = Intent.TEXT_RECOGNITION,
			confidence = probabilities[Intent.TEXT_RECOGNITION.ordinal],
			originalText = originalText,
			probabilities = probabilities
		)

		val encoded = PendingExternalIntentCodec.encode(PendingExternalIntent(original))
		val restored = PendingExternalIntentCodec.decode(encoded)?.intentResult

		assertEquals(original.intent, restored?.intent)
		assertEquals(original.confidence, restored?.confidence)
		assertEquals(originalText, restored?.originalText)
		assertArrayEquals(probabilities, restored?.probabilities, 0f)
	}

	@Test
	fun invalidOrNonPendingJsonCannotBecomeAnExternalCommand() {
		assertNull(PendingExternalIntentCodec.decode(null))
		assertNull(PendingExternalIntentCodec.decode("{}"))
		assertNull(PendingExternalIntentCodec.decode("not-json"))
	}

	@Test
	fun everyGlobalIntentGetsAnExplicitSettingsContextQuestion() {
		listOf(
			Intent.TEXT_RECOGNITION,
			Intent.OBJECT_DETECTION,
			Intent.MEASURE_DISTANCE,
			Intent.REDIRECT_TO_LLM
		).forEach { intent ->
			val question = PendingExternalIntentPresentation.confirmationQuestion(intent)
			assertTrue(question.startsWith("Sie befinden sich noch in den Einstellungen."))
			assertTrue(question.endsWith("?"))
		}
	}
}
