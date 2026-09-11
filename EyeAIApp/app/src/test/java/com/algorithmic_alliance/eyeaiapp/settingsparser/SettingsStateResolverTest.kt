package com.algorithmic_alliance.eyeaiapp.settingsparser

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SettingsStateResolverTest {
	private val resolver = SettingsStateResolver()

	private fun command(
		operation: SettingOperation, value: Double? = null, magnitude: ChangeMagnitude? = null
	) = SettingCommand(
		target = SettingTarget.FREQUENCY,
		operation = operation,
		operationConfidence = 1f,
		numericValue = value,
		magnitude = magnitude,
		speaker = null,
		speakerConfidence = null,
		unit = SettingUnit.HZ,
		status = SettingParseStatus.COMPLETE,
		originalText = "test",
		normalizedText = "test"
	)

	@Test
	fun `small increase resolves through central step configuration`() {
		val resolution = resolver.resolve(
			command(SettingOperation.INCREASE, magnitude = ChangeMagnitude.SMALL),
			CurrentSettingsState(600, 2.0, 1.0, SpeakerChoice.MALE)
		)
		assertEquals(SettingParseStatus.COMPLETE, resolution.status)
		assertEquals(650.0, (resolution.requestedValue as ResolvedSettingValue.Numeric).value, 0.0)
	}

	@Test
	fun `numeric delta resolves separately from absolute value`() {
		val resolution = resolver.resolve(
			command(SettingOperation.INCREASE, value = 100.0),
			CurrentSettingsState(600, 2.0, 1.0, SpeakerChoice.MALE)
		)
		assertEquals(700.0, (resolution.requestedValue as ResolvedSettingValue.Numeric).value, 0.0)
	}

	@Test
	fun `out of range final value is rejected rather than clamped`() {
		val resolution = resolver.resolve(
			command(SettingOperation.INCREASE, magnitude = ChangeMagnitude.LARGE),
			CurrentSettingsState(3900, 2.0, 1.0, SpeakerChoice.MALE)
		)
		assertEquals(SettingParseStatus.INVALID_VALUE, resolution.status)
		assertNull(resolution.requestedValue)
	}

	@Test
	fun `speaker toggle with unknown current speaker never selects a default`() {
		val speakerCommand = command(SettingOperation.TOGGLE).copy(
			target = SettingTarget.SPEAKER, speaker = SpeakerChoice.UNSPECIFIED, unit = null
		)
		val resolution = resolver.resolve(
			speakerCommand, CurrentSettingsState(600, 2.0, 1.0, SpeakerChoice.UNSPECIFIED)
		)
		assertEquals(SettingParseStatus.NEEDS_CLARIFICATION, resolution.status)
		assertNull(resolution.requestedValue)
	}

}
