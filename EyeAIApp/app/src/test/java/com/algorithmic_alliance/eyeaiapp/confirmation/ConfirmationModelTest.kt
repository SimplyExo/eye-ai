package com.algorithmic_alliance.eyeaiapp.confirmation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ConfirmationModelTest {
	private val model by lazy(ConfirmationModelTestFixture::load)

	@Test
	fun commonShortAnswersUseDeterministicRules() {
		val examples = mapOf(
			ConfirmationLabel.ACCEPT to listOf(
				"Ja.", "genau", "auf jeden Fall", "klingt gut", "mach weiter"
			),
			ConfirmationLabel.REJECT to listOf(
				"Nein.", "stopp", "abbrechen", "auf gar keinen Fall", "vergiss es"
			),
			ConfirmationLabel.UNKNOWN to listOf(
				"weiß nicht", "hm", "kommt drauf an", "ich bin unsicher", "noch nicht"
			)
		)
		examples.forEach { (expected, answers) ->
			answers.forEach { answer ->
				val result = model.classify(QUESTION_FREQUENCY, answer, ACTION_FREQUENCY)
				assertEquals(answer, expected, result.label)
				assertEquals(answer, "fast_rule", result.source)
				assertEquals(answer, 1.0, result.confidence, 0.0)
			}
		}
	}

	@Test
	fun decisionTraceFieldsExposeLocalDecisionAndAllScores() {
		val accepted = model.classify(QUESTION_FREQUENCY, "Ja.", ACTION_FREQUENCY)
		val trace = accepted.toDecisionTraceFields()

		assertTrue(trace.contains("model=deterministic_char_ngram_v1"))
		assertTrue(trace.contains("decision=ACCEPT confirmed=true"))
		assertTrue(trace.contains("requiresClarification=false"))
		assertTrue(trace.contains("confidence=1.0000 threshold=0.4000"))
		assertTrue(trace.contains("source=fast_rule reason=deterministic_exact_phrase"))
		assertTrue(trace.contains("scores=[ACCEPT=1.0000, REJECT=0.0000, UNKNOWN=0.0000]"))
	}

	@Test
	fun androidInferenceMatchesFrozenSklearnProbabilities() {
		val cases = listOf(
			ParityCase(
				question = QUESTION_FREQUENCY,
				answer = "Ja, diese Anpassung am Signalton ist gewollt.",
				pendingAction = ACTION_FREQUENCY,
				expectedLabel = ConfirmationLabel.ACCEPT,
				expectedScores = doubleArrayOf(
					0.6673445232598857,
					0.14911717954675985,
					0.1835382971933545
				)
			),
			ParityCase(
				question = "Verstanden. Soll ich die BPS auf 4.0 setzen?",
				answer = "Die neue BPS-Einstellung möchte ich nicht übernehmen.",
				pendingAction = "die BPS auf 4.0 setzen",
				expectedLabel = ConfirmationLabel.REJECT,
				expectedScores = doubleArrayOf(
					0.21652127261776946,
					0.6603168998255656,
					0.12316182755666504
				)
			),
			ParityCase(
				question = "Verstanden. Soll die Assistentenstimme nun männlich sein?",
				answer = "Welche Stimme ist gerade aktiv?",
				pendingAction = "zur männlichen Assistentenstimme wechseln",
				expectedLabel = ConfirmationLabel.UNKNOWN,
				expectedScores = doubleArrayOf(
					0.07925037063787671,
					0.05348262080018633,
					0.867267008561937
				)
			)
		)

		cases.forEach { case ->
			val result = model.classify(case.question, case.answer, case.pendingAction)
			assertEquals(case.answer, case.expectedLabel, result.label)
			ConfirmationLabel.entries.forEachIndexed { index, label ->
				assertEquals(
					"${case.answer}: $label",
					case.expectedScores[index],
					result.scores.getValue(label),
					1e-12
				)
			}
		}
	}

	@Test
	fun lowConfidenceActionablePredictionBecomesUnknownWithoutChangingRawLabel() {
		val result = model.classify(
			question = "Sie befinden sich noch in den Einstellungen. Möchten Sie die " +
				"Einstellungen verlassen und die Entfernung messen?",
			answer = "herr lasse das einstellung menü nicht für die entfernungsmessung",
			pendingAction = "die Einstellungen verlassen und die Entfernung messen"
		)

		assertEquals(ConfirmationLabel.UNKNOWN, result.label)
		assertEquals(ConfirmationLabel.REJECT, result.rawLabel)
		assertEquals("low_char_confidence", result.decisionReason)
		assertEquals("char_ngram_confidence_reject", result.source)
		assertEquals(0.3969747999842313, result.confidence, 1e-12)
		assertTrue(result.confidence < 0.40)
		val trace = result.toDecisionTraceFields()
		assertTrue(trace.contains("decision=UNKNOWN"))
		assertTrue(trace.contains("confirmed=false rejected=false requiresClarification=true"))
		assertTrue(trace.contains("rawLabel=REJECT"))
		assertTrue(trace.contains("confidenceRejected=true"))
	}

	private data class ParityCase(
		val question: String,
		val answer: String,
		val pendingAction: String,
		val expectedLabel: ConfirmationLabel,
		val expectedScores: DoubleArray
	)

	companion object {
		private const val QUESTION_FREQUENCY =
			"Verstanden. Soll ich die Audio-Frequenz auf 700 Hz setzen?"
		private const val ACTION_FREQUENCY = "die Audio-Frequenz auf 700 Hz setzen"
	}
}
