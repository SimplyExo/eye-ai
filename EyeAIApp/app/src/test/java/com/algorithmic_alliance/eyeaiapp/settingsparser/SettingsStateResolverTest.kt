package com.algorithmic_alliance.eyeaiapp.settingsparser

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.json.JSONObject

class SettingsStateResolverTest {
	private val resolver = SettingsStateResolver()

	private fun command(
		operation: SettingOperation,
		value: Double? = null,
		magnitude: ChangeMagnitude? = null
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
			target = SettingTarget.SPEAKER,
			speaker = SpeakerChoice.UNSPECIFIED,
			unit = null
		)
		val resolution = resolver.resolve(
			speakerCommand,
			CurrentSettingsState(600, 2.0, 1.0, SpeakerChoice.UNSPECIFIED)
		)
		assertEquals(SettingParseStatus.NEEDS_CLARIFICATION, resolution.status)
		assertNull(resolution.requestedValue)
	}

	@Test
	fun `shared Python Kotlin state resolution golden cases match`() {
		val cases = JSONObject(
			requireNotNull(javaClass.getResource("/state_resolution_golden.json")).readText()
		).getJSONArray("cases")
		for (index in 0 until cases.length()) {
			val item = cases.getJSONObject(index)
			val current = item.getJSONObject("current")
			val unit = if (item.isNull("unit")) null else SettingUnit.valueOf(item.getString("unit"))
			val result = resolver.resolve(
				SettingCommand(
					target = SettingTarget.valueOf(item.getString("target")),
					operation = SettingOperation.valueOf(item.getString("operation")),
					operationConfidence = 1f,
					numericValue = if (item.isNull("numeric_value")) null else item.getDouble("numeric_value"),
					magnitude = if (item.isNull("magnitude")) null else ChangeMagnitude.valueOf(item.getString("magnitude")),
					speaker = if (item.isNull("speaker")) null else SpeakerChoice.valueOf(item.getString("speaker")),
					speakerConfidence = null,
					unit = unit,
					status = SettingParseStatus.COMPLETE,
					originalText = "golden state",
					normalizedText = "golden state"
				),
				CurrentSettingsState(
					current.getInt("frequency"),
					current.getDouble("bps"),
					current.getDouble("speech_speed"),
					SpeakerChoice.valueOf(current.getString("speaker"))
				)
			)
			assertEquals(
				item.getString("name"),
				SettingParseStatus.valueOf(item.getString("expected_status")),
				result.status
			)
			when {
				!item.isNull("expected_numeric_value") -> assertEquals(
					item.getDouble("expected_numeric_value"),
					(result.requestedValue as ResolvedSettingValue.Numeric).value,
					1e-9
				)
				!item.isNull("expected_speaker") -> assertEquals(
					SpeakerChoice.valueOf(item.getString("expected_speaker")),
					(result.requestedValue as ResolvedSettingValue.Speaker).value
				)
				else -> assertNull(result.requestedValue)
			}
		}
	}
}
