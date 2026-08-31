package com.algorithmic_alliance.eyeaiapp.settingsparser

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

class SettingCommandRuleMatrixTest {
	@Test
	fun `all shared 4 by 5 matrix and edge cases match Kotlin validator`() {
		val payload = JSONObject(requireNotNull(javaClass.getResource("/command_rule_matrix.json")).readText())
		val cases = payload.getJSONArray("cases")
		val primaryPairs = (0 until 20).map { index ->
			val item = cases.getJSONObject(index)
			item.getString("target") to item.getString("operation")
		}.toSet()
		assertEquals(
			SettingTarget.entries.flatMap { target ->
				SettingOperation.entries.map { operation -> target.name to operation.name }
			}.toSet(),
			primaryPairs
		)

		val validator = SettingCommandValidator()
		for (index in 0 until cases.length()) {
			val item = cases.getJSONObject(index)
			val command = SettingCommand(
				target = SettingTarget.valueOf(item.getString("target")),
				operation = SettingOperation.valueOf(item.getString("operation")),
				operationConfidence = 1f,
				numericValue = if (item.isNull("numeric_value")) null else item.getDouble("numeric_value"),
				magnitude = if (item.isNull("magnitude")) null else ChangeMagnitude.valueOf(item.getString("magnitude")),
				speaker = if (item.isNull("speaker")) null else SpeakerChoice.valueOf(item.getString("speaker")),
				speakerConfidence = null,
				unit = if (item.isNull("unit")) null else SettingUnit.valueOf(item.getString("unit")),
				status = SettingParseStatus.COMPLETE,
				originalText = "golden matrix",
				normalizedText = "golden matrix"
			)
			assertEquals(
				item.getString("name"),
				SettingParseStatus.valueOf(item.getString("expected_status")),
				validator.validate(command).status
			)
		}
	}
}
