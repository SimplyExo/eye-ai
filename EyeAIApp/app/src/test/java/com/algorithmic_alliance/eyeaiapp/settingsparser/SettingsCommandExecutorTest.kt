package com.algorithmic_alliance.eyeaiapp.settingsparser

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SettingsCommandExecutorTest {
	private val executor = SettingsCommandExecutor()
	private val current = CurrentSettingsState(600, 2.0, 1.0, SpeakerChoice.MALE)

	@Test
	fun `validated local command is translated into existing confirmation JSON`() {
		val execution = executor.execute(
			command(SettingTarget.FREQUENCY, SettingOperation.SET_ABSOLUTE, numeric = 700.0, unit = SettingUnit.HZ),
			current
		)
		assertTrue(execution is LocalSettingsCommandExecution.Ready)
		val json = JSONObject((execution as LocalSettingsCommandExecution.Ready).settingsJson)
		assertTrue(json.getBoolean("settings_parameter_complete"))
		assertEquals(700, json.getJSONArray("changed_settings").getJSONObject(0).getInt("frequency"))
	}

	@Test
	fun `relative command is resolved before existing confirmation JSON is created`() {
		val execution = executor.execute(
			command(SettingTarget.FREQUENCY, SettingOperation.INCREASE, magnitude = ChangeMagnitude.SMALL, unit = SettingUnit.HZ),
			current
		) as LocalSettingsCommandExecution.Ready
		assertEquals(650, JSONObject(execution.settingsJson).getJSONArray("changed_settings").getJSONObject(0).getInt("frequency"))
	}

	@Test
	fun `speaker toggle uses existing voice integer schema after state resolution`() {
		val execution = executor.execute(
			command(
				SettingTarget.SPEAKER,
				SettingOperation.TOGGLE,
				speaker = SpeakerChoice.UNSPECIFIED,
				unit = null
			),
			current
		) as LocalSettingsCommandExecution.Ready
		assertEquals(0, JSONObject(execution.settingsJson).getJSONArray("changed_settings").getJSONObject(0).getInt("voice"))
	}

	@Test
	fun `fractional valid BPS command remains a parser command but is explicit at Android boundary`() {
		val execution = executor.execute(
			command(SettingTarget.BPS, SettingOperation.SET_ABSOLUTE, numeric = 5.5, unit = SettingUnit.BPS),
			current
		)
		assertTrue(execution is LocalSettingsCommandExecution.UnsupportedAppRepresentation)
		assertEquals(
			"ANDROID_BPS_REQUIRES_INTEGER",
			(execution as LocalSettingsCommandExecution.UnsupportedAppRepresentation).diagnostic
		)
	}

	private fun command(
		target: SettingTarget,
		operation: SettingOperation,
		numeric: Double? = null,
		magnitude: ChangeMagnitude? = null,
		speaker: SpeakerChoice? = null,
		unit: SettingUnit?
	) = SettingCommand(
		target = target,
		operation = operation,
		operationConfidence = 1f,
		numericValue = numeric,
		magnitude = magnitude,
		speaker = speaker,
		speakerConfidence = if (speaker == null) null else 1f,
		unit = unit,
		status = SettingParseStatus.COMPLETE,
		originalText = "test",
		normalizedText = "test"
	)
}
