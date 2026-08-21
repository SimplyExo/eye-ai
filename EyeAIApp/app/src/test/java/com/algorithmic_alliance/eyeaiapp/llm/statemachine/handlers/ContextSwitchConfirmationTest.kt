package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModelTestFixture
import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import com.algorithmic_alliance.eyeaiapp.nlp.IntentResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextSwitchConfirmationTest {
	private val model by lazy(ConfirmationModelTestFixture::load)
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
	fun approvalIsEvaluatedLocally() {
		val traces = mutableListOf<String>()
		val confirmation = ContextSwitchConfirmation({ model }, traces::add)

		assertEquals(
			ContextSwitchConfirmationResult.APPROVED,
			confirmation.evaluate("Ja.", pending)
		)
		assertTrue(traces.any { it.contains("[ConfirmationModel][EVALUATE]") })
		assertTrue(traces.any { it.contains("evaluator=LOCAL_CONFIRMATION_MODEL") })
		assertTrue(traces.any { it.contains("apiCalled=false") })
		assertTrue(traces.any { it.contains("decision=ACCEPT") && it.contains("confirmed=true") })
		assertTrue(traces.any { it.contains("scores=[ACCEPT=1.0000, REJECT=0.0000, UNKNOWN=0.0000]") })
		assertTrue(traces.none { it.contains("Gemini") })
	}

	@Test
	fun rejectionDoesNotApproveStoredAction() {
		val confirmation = ContextSwitchConfirmation({ model })

		assertEquals(
			ContextSwitchConfirmationResult.REJECTED,
			confirmation.evaluate("Nein.", pending)
		)
	}

	@Test
	fun unknownKeepsDecisionUnresolved() {
		val confirmation = ContextSwitchConfirmation({ model })

		assertEquals(
			ContextSwitchConfirmationResult.UNKNOWN,
			confirmation.evaluate("Ich bin unsicher.", pending)
		)
	}

	@Test
	fun explicitAbortRemainsSeparateFromTheThreeModelLabels() {
		var modelLoads = 0
		val confirmation = ContextSwitchConfirmation({
			modelLoads++
			model
		})

		assertEquals(
			ContextSwitchConfirmationResult.ABORTED,
			confirmation.evaluate("Abbrechen.", pending)
		)
		assertEquals(0, modelLoads)
	}
}
