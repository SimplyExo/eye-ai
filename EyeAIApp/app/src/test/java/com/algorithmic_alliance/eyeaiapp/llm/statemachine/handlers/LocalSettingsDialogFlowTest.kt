package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.settingsparser.CurrentSettingsState
import com.algorithmic_alliance.eyeaiapp.settingsparser.LocalSettingsParser
import com.algorithmic_alliance.eyeaiapp.settingsparser.OperationPrediction
import com.algorithmic_alliance.eyeaiapp.settingsparser.OperationPredictor
import com.algorithmic_alliance.eyeaiapp.settingsparser.SettingOperation
import com.algorithmic_alliance.eyeaiapp.settingsparser.SettingParseStatus
import com.algorithmic_alliance.eyeaiapp.settingsparser.SettingTarget
import com.algorithmic_alliance.eyeaiapp.settingsparser.SpeakerChoice
import com.algorithmic_alliance.eyeaiapp.settingsparser.SpeakerPrediction
import com.algorithmic_alliance.eyeaiapp.settingsparser.SpeakerPredictor
import com.algorithmic_alliance.eyeaiapp.settingsparser.Text2NumGermanNumberNormalizer
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalSettingsDialogFlowTest {
	private val jsonParser = JsonParser()
	private val flow = LocalSettingsDialogFlow(jsonParser)
	private val currentState = CurrentSettingsState(
		frequency = 600,
		bps = 2.0,
		speechSpeed = 1.0,
		speaker = SpeakerChoice.FEMALE
	)

	@Test
	fun `complete command enters the existing local confirmation path`() {
		val context = jsonParser.createSettingsContext(
			SettingIntent.FREQUENCY,
			SettingsFlow.DIRECT,
			"Setze die Frequenz auf 700"
		)

		val result = flow.process(
			input = "700",
			currentJson = context,
			currentState = currentState,
			parser = parser(SettingOperation.SET_ABSOLUTE)
		)

		assertTrue(result is LocalSettingsDialogResult.Ready)
		result as LocalSettingsDialogResult.Ready
		assertEquals(SettingTarget.FREQUENCY, result.execution.command.target)
		assertEquals(SettingParseStatus.COMPLETE, result.execution.command.status)
		assertEquals(700, JSONObject(result.confirmationJson)
			.getJSONArray("changed_settings").getJSONObject(0).getInt("frequency"))
		assertEquals(SettingsFlow.DIRECT, jsonParser.parseSettingsFlow(result.confirmationJson))
	}

	@Test
	fun `needs value retains target context and completes a short numeric follow up locally`() {
		val context = jsonParser.createSettingsContext(SettingIntent.FREQUENCY)
		val missing = flow.process(
			input = "Frequenz einstellen",
			currentJson = context,
			currentState = currentState,
			parser = parser(SettingOperation.SET_ABSOLUTE)
		)

		assertTrue(missing is LocalSettingsDialogResult.FollowUp)
		missing as LocalSettingsDialogResult.FollowUp
		assertEquals(SettingParseStatus.NEEDS_VALUE, missing.status)
		assertEquals(context, missing.retainedContextJson)

		val parsedTargets = mutableListOf<SettingTarget>()
		val complete = flow.process(
			input = "120",
			currentJson = missing.retainedContextJson,
			currentState = currentState,
			parser = parser(SettingOperation.SET_ABSOLUTE, parsedTargets = parsedTargets)
		)

		assertTrue(complete is LocalSettingsDialogResult.Ready)
		complete as LocalSettingsDialogResult.Ready
		assertEquals(listOf(SettingTarget.FREQUENCY), parsedTargets)
		assertEquals(120, JSONObject(complete.confirmationJson)
			.getJSONArray("changed_settings").getJSONObject(0).getInt("frequency"))
	}

	@Test
	fun `known frequency target asks only for the missing operation`() {
		val context = jsonParser.createSettingsContext(SettingIntent.FREQUENCY)

		val result = flow.process(
			input = "Ändere die Frequenz",
			currentJson = context,
			currentState = currentState,
			parser = parser(SettingOperation.UNSPECIFIED)
		) as LocalSettingsDialogResult.FollowUp

		assertEquals(
			"Verstanden, Sie möchten die Frequenz der Distanzhinweistöne anpassen. " +
				"Möchten Sie die Frequenz erhöhen, verringern oder einen konkreten Wert einstellen?",
			result.question
		)
		assertEquals(context, result.retainedContextJson)
		assertTrue(result.question.contains("Frequenz"))
		assertTrue(!result.question.contains("Mit dieser Einstellung"))
	}

	@Test
	fun `missing operation questions are derived from the known target`() {
		SettingIntent.entries
			.filter { it != SettingIntent.LEAVE && it != SettingIntent.NONE }
			.forEach { settingIntent ->
				val context = jsonParser.createSettingsContext(settingIntent)
				val result = flow.process(
					input = "Ändere die Einstellung",
					currentJson = context,
					currentState = currentState,
					parser = parser(SettingOperation.UNSPECIFIED)
				) as LocalSettingsDialogResult.FollowUp

				assertEquals(settingIntent.missingOperationQuestion(), result.question)
				assertEquals(context, result.retainedContextJson)
			}
	}

	@Test
	fun `needs clarification retains context and completes a short speaker answer locally`() {
		val context = jsonParser.createSettingsContext(SettingIntent.VOICE)
		val clarification = flow.process(
			input = "wechsle die Stimme und nimm die weibliche",
			currentJson = context,
			currentState = currentState,
			parser = parser(SettingOperation.TOGGLE, SpeakerChoice.FEMALE)
		)

		assertTrue(clarification is LocalSettingsDialogResult.FollowUp)
		clarification as LocalSettingsDialogResult.FollowUp
		assertEquals(SettingParseStatus.NEEDS_CLARIFICATION, clarification.status)
		assertEquals(context, clarification.retainedContextJson)

		val complete = flow.process(
			input = "männlich",
			currentJson = clarification.retainedContextJson,
			currentState = currentState,
			parser = parser(SettingOperation.SET_ABSOLUTE, SpeakerChoice.MALE)
		)

		assertTrue(complete is LocalSettingsDialogResult.Ready)
		complete as LocalSettingsDialogResult.Ready
		assertEquals(1, JSONObject(complete.confirmationJson)
			.getJSONArray("changed_settings").getJSONObject(0).getInt("voice"))
	}

	@Test
	fun `short relative follow ups retain their frequency target`() {
		val context = jsonParser.createSettingsContext(SettingIntent.FREQUENCY)
		val cases = listOf(
			"mehr" to (SettingOperation.INCREASE to 700),
			"weniger" to (SettingOperation.DECREASE to 500),
			"erhöhen" to (SettingOperation.INCREASE to 700),
			"verringern" to (SettingOperation.DECREASE to 500),
			"120 Hertz" to (SettingOperation.SET_ABSOLUTE to 120)
		)

		for ((input, expectation) in cases) {
			val result = flow.process(
				input = input,
				currentJson = context,
				currentState = currentState,
				parser = parser(expectation.first)
			)

			assertTrue(input, result is LocalSettingsDialogResult.Ready)
			result as LocalSettingsDialogResult.Ready
			assertEquals(
				input,
				expectation.second,
				JSONObject(result.confirmationJson)
					.getJSONArray("changed_settings").getJSONObject(0).getInt("frequency")
			)
		}
	}

	@Test
	fun `invalid values and units stay in the local correction loop`() {
		val context = jsonParser.createSettingsContext(SettingIntent.FREQUENCY)
		val invalidValue = flow.process(
			input = "50 Hertz",
			currentJson = context,
			currentState = currentState,
			parser = parser(SettingOperation.SET_ABSOLUTE)
		) as LocalSettingsDialogResult.FollowUp
		assertEquals(SettingParseStatus.INVALID_VALUE, invalidValue.status)
		assertEquals(context, invalidValue.retainedContextJson)

		val invalidUnit = flow.process(
			input = "700 BPS",
			currentJson = context,
			currentState = currentState,
			parser = parser(SettingOperation.SET_ABSOLUTE)
		) as LocalSettingsDialogResult.FollowUp
		assertEquals(SettingParseStatus.INVALID_UNIT, invalidUnit.status)
		assertEquals(context, invalidUnit.retainedContextJson)
	}

	@Test
	fun `unavailable local parser requests a retained local retry`() {
		val context = jsonParser.createSettingsContext(SettingIntent.FREQUENCY)

		val result = flow.process(
			input = "700",
			currentJson = context,
			currentState = currentState,
			parser = null
		)

		assertTrue(result is LocalSettingsDialogResult.FollowUp)
		result as LocalSettingsDialogResult.FollowUp
		assertEquals("LOCAL_RUNTIME_UNAVAILABLE", result.diagnostic)
		assertEquals(context, result.retainedContextJson)
		assertTrue(result.question.contains("lokale Verarbeitung"))
	}

	private fun parser(
		operation: SettingOperation,
		speaker: SpeakerChoice = SpeakerChoice.UNSPECIFIED,
		parsedTargets: MutableList<SettingTarget>? = null
	): LocalSettingsParser = LocalSettingsParser(
		numberNormalizer = Text2NumGermanNumberNormalizer(),
		operationPredictor = object : OperationPredictor {
			override fun predictOperation(
				target: SettingTarget,
				normalizedText: String
			): OperationPrediction {
				parsedTargets?.add(target)
				return OperationPrediction(operation, 1f)
			}
		},
		speakerPredictor = object : SpeakerPredictor {
			override fun predictSpeaker(
				target: SettingTarget,
				normalizedText: String
			): SpeakerPrediction = SpeakerPrediction(speaker, 1f)
		}
	)
}
