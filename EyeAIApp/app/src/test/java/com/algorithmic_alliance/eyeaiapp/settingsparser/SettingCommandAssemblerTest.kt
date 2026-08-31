package com.algorithmic_alliance.eyeaiapp.settingsparser

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SettingCommandAssemblerTest {
	private val assembler = SettingCommandAssembler()
	private val validator = SettingCommandValidator()
	private val units = SettingUnitParser()

	private fun operation(operation: SettingOperation) = OperationPrediction(operation, 0.9f)

	private fun numbers(vararg values: Double): NumberNormalizationResult = NumberNormalizationResult(
		originalText = "test",
		normalizedText = if (values.isEmpty()) "test" else values.joinToString(" ") { "<NUM>" },
		values = values.toList(),
		occurrences = values.map {
			NumberOccurrence(it.toString(), it, status = NumberOccurrenceStatus.SUCCESS)
		},
		status = when (values.size) {
			0 -> NumberNormalizationStatus.NO_NUMBER
			1 -> NumberNormalizationStatus.SUCCESS
			else -> NumberNormalizationStatus.AMBIGUOUS
		},
		normalizerId = "test",
		normalizerVersion = "1"
	)

	private fun magnitude(value: ChangeMagnitude?): MagnitudeParseResult =
		MagnitudeParseResult(value, MagnitudeParseStatus.CLEAR)

	@Test
	fun `absolute value missing value and unspecified operation use canonical statuses`() {
		val complete = validator.validate(
			assembler.assemble(
				SettingTarget.FREQUENCY,
				"setze sie auf 700",
				numbers(700.0),
				operation(SettingOperation.SET_ABSOLUTE),
				null,
				magnitude(null),
				units.parse(SettingTarget.FREQUENCY, "setze sie auf 700")
			)
		)
		assertEquals(700.0, complete.numericValue)
		assertEquals(SettingParseStatus.COMPLETE, complete.status)

		val missing = validator.validate(
			assembler.assemble(
				SettingTarget.FREQUENCY,
				"setze die frequenz",
				numbers(),
				operation(SettingOperation.SET_ABSOLUTE),
				null,
				magnitude(null),
				units.parse(SettingTarget.FREQUENCY, "setze die frequenz")
			)
		)
		assertEquals(SettingParseStatus.NEEDS_VALUE, missing.status)

		val unspecified = validator.validate(missing.copy(operation = SettingOperation.UNSPECIFIED))
		assertEquals(SettingParseStatus.NEEDS_CLARIFICATION, unspecified.status)
	}

	@Test
	fun `multiple numbers remain diagnostic and never become numeric value`() {
		val command = validator.validate(
			assembler.assemble(
				SettingTarget.FREQUENCY,
				"von 600 auf 700",
				numbers(600.0, 700.0),
				operation(SettingOperation.SET_ABSOLUTE),
				null,
				magnitude(null),
				units.parse(SettingTarget.FREQUENCY, "von 600 auf 700")
			)
		)
		assertNull(command.numericValue)
		assertEquals(listOf(600.0, 700.0), command.extractedNumericValues)
		assertEquals(SettingParseStatus.NEEDS_CLARIFICATION, command.status)
	}

	@Test
	fun `speaker toggle and explicit invalid unit propagate correctly`() {
		val toggle = validator.validate(
			assembler.assemble(
				SettingTarget.SPEAKER,
				"nimm eine andere stimme",
				numbers(),
				operation(SettingOperation.TOGGLE),
				SpeakerPrediction(SpeakerChoice.UNSPECIFIED, 0.8f),
				magnitude(null),
				units.parse(SettingTarget.SPEAKER, "nimm eine andere stimme")
			)
		)
		assertEquals(SettingParseStatus.COMPLETE, toggle.status)
		assertEquals(SpeakerChoice.UNSPECIFIED, toggle.speaker)

		val invalidUnit = assembler.assemble(
			SettingTarget.FREQUENCY,
			"setze sie auf 700 bps",
			numbers(700.0),
			operation(SettingOperation.SET_ABSOLUTE),
			null,
			magnitude(null),
			units.parse(SettingTarget.FREQUENCY, "setze sie auf 700 bps")
		)
		assertEquals(SettingParseStatus.INVALID_UNIT, invalidUnit.status)
	}
}
