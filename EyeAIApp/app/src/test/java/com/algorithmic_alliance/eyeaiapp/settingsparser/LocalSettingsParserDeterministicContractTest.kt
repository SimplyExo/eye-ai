package com.algorithmic_alliance.eyeaiapp.settingsparser

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalSettingsParserDeterministicContractTest {
	@Test
	fun `absolute set and relative increase decrease use deterministic number unit assembly`() {
		val absolute = parser(SettingOperation.SET_ABSOLUTE).parse(
			SettingTarget.FREQUENCY,
			"setze die frequenz auf siebenhundert hertz"
		)
		assertCompleteNumeric(absolute, SettingOperation.SET_ABSOLUTE, 700.0)

		val increase = parser(SettingOperation.INCREASE).parse(
			SettingTarget.FREQUENCY,
			"erhöhe die frequenz um einhundert hertz"
		)
		assertCompleteNumeric(increase, SettingOperation.INCREASE, 100.0)

		val decrease = parser(SettingOperation.DECREASE).parse(
			SettingTarget.FREQUENCY,
			"senke die frequenz um einhundert hertz"
		)
		assertCompleteNumeric(decrease, SettingOperation.DECREASE, 100.0)
	}

	@Test
	fun `minus becomes a positive delta only for lexical decrease`() {
		val decrease = parser(SettingOperation.DECREASE).parse(
			SettingTarget.FREQUENCY,
			"minus einhundert hertz"
		)
		assertCompleteNumeric(decrease, SettingOperation.DECREASE, 100.0)
		assertEquals(listOf(100.0), decrease.extractedNumericValues)
		assertEquals(-100.0, decrease.numberOccurrences.single().value)

		val absolute = parser(SettingOperation.SET_ABSOLUTE).parse(
			SettingTarget.FREQUENCY,
			"minus einhundert hertz"
		)
		assertEquals(-100.0, absolute.numericValue)
		assertEquals(SettingParseStatus.INVALID_VALUE, absolute.status)

		val increase = parser(SettingOperation.INCREASE).parse(
			SettingTarget.FREQUENCY,
			"minus einhundert hertz"
		)
		assertEquals(-100.0, increase.numericValue)
		assertEquals(SettingParseStatus.INVALID_VALUE, increase.status)

		val symbolic = parser(SettingOperation.DECREASE).parse(
			SettingTarget.FREQUENCY,
			"-100 hertz"
		)
		assertEquals(-100.0, symbolic.numericValue)
		assertEquals(SettingParseStatus.INVALID_VALUE, symbolic.status)
	}

	@Test
	fun `short mehr weniger rauf runter commands receive deterministic default deltas`() {
		listOf(
			"mehr" to SettingOperation.INCREASE,
			"weniger" to SettingOperation.DECREASE,
			"rauf" to SettingOperation.INCREASE,
			"runter" to SettingOperation.DECREASE
		).forEach { (text, operation) ->
			val command = parser(operation).parse(SettingTarget.FREQUENCY, text)
			assertEquals(text, SettingParseStatus.COMPLETE, command.status)
			assertEquals(text, ChangeMagnitude.DEFAULT, command.magnitude)
			assertNull(text, command.numericValue)
			assertEquals(text, SettingUnit.HZ, command.unit)
		}
	}

	@Test
	fun `missing value and invalid numeric or unit remain explicit statuses`() {
		val missing = parser(SettingOperation.SET_ABSOLUTE).parse(
			SettingTarget.FREQUENCY,
			"für die frequenz bitte einen wert einstellen"
		)
		assertEquals(SettingParseStatus.NEEDS_VALUE, missing.status)
		assertTrue(missing.diagnostics.contains("NUMERIC_VALUE_REQUIRED"))

		val wrongUnit = parser(SettingOperation.SET_ABSOLUTE).parse(
			SettingTarget.FREQUENCY,
			"setze die frequenz auf siebenhundert bps"
		)
		assertEquals(SettingParseStatus.INVALID_UNIT, wrongUnit.status)

		val wrongValue = parser(SettingOperation.SET_ABSOLUTE).parse(
			SettingTarget.FREQUENCY,
			"setze die frequenz auf fünfzig hertz"
		)
		assertEquals(SettingParseStatus.INVALID_VALUE, wrongValue.status)
	}

	@Test
	fun `speaker choice toggle and explicit toggle choice follow validator contract`() {
		val male = parser(SettingOperation.SET_ABSOLUTE, SpeakerChoice.MALE).parse(SettingTarget.SPEAKER, "maskulin")
		assertEquals(SettingParseStatus.COMPLETE, male.status)
		assertEquals(SpeakerChoice.MALE, male.speaker)

		val female = parser(SettingOperation.SET_ABSOLUTE, SpeakerChoice.FEMALE).parse(SettingTarget.SPEAKER, "feminin")
		assertEquals(SettingParseStatus.COMPLETE, female.status)
		assertEquals(SpeakerChoice.FEMALE, female.speaker)

		val toggle = parser(SettingOperation.TOGGLE, SpeakerChoice.UNSPECIFIED).parse(
			SettingTarget.SPEAKER,
			"auf eine andere stimmenausgabe umsteigen"
		)
		assertEquals(SettingParseStatus.COMPLETE, toggle.status)
		assertEquals(SpeakerChoice.UNSPECIFIED, toggle.speaker)

		val toggleWithChoice = parser(SettingOperation.TOGGLE, SpeakerChoice.FEMALE).parse(
			SettingTarget.SPEAKER,
			"wechsel die stimme und nimm die feminine"
		)
		assertEquals(SettingParseStatus.NEEDS_CLARIFICATION, toggleWithChoice.status)
		assertTrue(toggleWithChoice.diagnostics.contains("TOGGLE_SPEAKER_MUST_BE_UNSPECIFIED"))
	}

	@Test
	fun `status output is deterministic across repeated local parses`() {
		val parser = parser(SettingOperation.INCREASE)
		val first = parser.parse(SettingTarget.FREQUENCY, "noch etwas rauf")
		repeat(5) {
			val again = parser.parse(SettingTarget.FREQUENCY, "noch etwas rauf")
			assertEquals(first.target, again.target)
			assertEquals(first.operation, again.operation)
			assertEquals(first.numericValue, again.numericValue)
			assertEquals(first.magnitude, again.magnitude)
			assertEquals(first.unit, again.unit)
			assertEquals(first.status, again.status)
			assertEquals(first.diagnostics, again.diagnostics)
			assertEquals(first.normalizedText, again.normalizedText)
		}
	}

	private fun parser(
		operation: SettingOperation,
		speaker: SpeakerChoice = SpeakerChoice.UNSPECIFIED
	): LocalSettingsParser = LocalSettingsParser(
		numberNormalizer = Text2NumGermanNumberNormalizer(),
		operationPredictor = object : OperationPredictor {
			override fun predictOperation(target: SettingTarget, normalizedText: String): OperationPrediction =
				OperationPrediction(operation, 0.99f)
		},
		speakerPredictor = object : SpeakerPredictor {
			override fun predictSpeaker(target: SettingTarget, normalizedText: String): SpeakerPrediction =
				SpeakerPrediction(speaker, 0.99f)
		}
	)

	private fun assertCompleteNumeric(command: SettingCommand, operation: SettingOperation, value: Double) {
		assertEquals(operation, command.operation)
		assertEquals(value, command.numericValue)
		assertEquals(SettingUnit.HZ, command.unit)
		assertEquals(SettingParseStatus.COMPLETE, command.status)
	}
}
