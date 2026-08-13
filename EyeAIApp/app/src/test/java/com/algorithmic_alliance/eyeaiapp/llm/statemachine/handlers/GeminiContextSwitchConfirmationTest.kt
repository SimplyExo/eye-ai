package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import com.algorithmic_alliance.eyeaiapp.nlp.IntentResult
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GeminiContextSwitchConfirmationTest {
	private val parser = JsonParser()
	private val pending = PendingExternalIntent(
		IntentResult(
			intent = Intent.TEXT_RECOGNITION,
			confidence = 1f,
			originalText = "Lies das Schild.",
			probabilities = FloatArray(Intent.CLASS_ORDER.size).apply {
				this[Intent.TEXT_RECOGNITION.ordinal] = 1f
			}
		)
	)

	@Test
	fun approvalIsEvaluatedByExactlyOneGeminiRequestUsingStoredOriginalText() = runBlocking {
		var requestCount = 0
		var capturedPrompt = ""
		val traces = mutableListOf<String>()
		val confirmation = GeminiContextSwitchConfirmation(
			jsonParser = parser,
			trace = traces::add
		) { prompt, structured ->
			requestCount++
			capturedPrompt = prompt
			assertTrue(structured)
			"""{"approval":1}"""
		}

		val result = confirmation.evaluate("Ja.", pending)

		assertEquals(ContextSwitchConfirmationResult.APPROVED, result)
		assertEquals(1, requestCount)
		assertTrue(capturedPrompt.contains("Lies das Schild."))
		assertTrue(traces.any { it.contains("outcome=APPROVED") })
	}

	@Test
	fun rejectionUsesGeminiAndDoesNotApproveStoredAction() = runBlocking {
		var requestCount = 0
		val confirmation = GeminiContextSwitchConfirmation(parser) { _, _ ->
			requestCount++
			"""{"approval":0}"""
		}

		val result = confirmation.evaluate("Nein.", pending)

		assertEquals(ContextSwitchConfirmationResult.REJECTED, result)
		assertEquals(1, requestCount)
	}

	@Test
	fun missingGeminiResponseKeepsDecisionUnresolved() = runBlocking {
		val confirmation = GeminiContextSwitchConfirmation(parser) { _, _ -> null }

		assertEquals(
			ContextSwitchConfirmationResult.FAILED,
			confirmation.evaluate("Ja.", pending)
		)
	}

	@Test
	fun malformedGeminiResponseIsNotTreatedAsUserRejection() = runBlocking {
		val confirmation = GeminiContextSwitchConfirmation(parser) { _, _ ->
			"""{"interaction_text":"Gemini API error"}"""
		}

		assertEquals(
			ContextSwitchConfirmationResult.FAILED,
			confirmation.evaluate("Ja.", pending)
		)
	}

	@Test
	fun explicitAbortIsNotCollapsedIntoOrdinaryNo() = runBlocking {
		val confirmation = GeminiContextSwitchConfirmation(parser) { _, _ ->
			"""{"approval":0,"abort_settings_flow":true}"""
		}

		assertEquals(
			ContextSwitchConfirmationResult.ABORTED,
			confirmation.evaluate("Abbrechen.", pending)
		)
	}
}
