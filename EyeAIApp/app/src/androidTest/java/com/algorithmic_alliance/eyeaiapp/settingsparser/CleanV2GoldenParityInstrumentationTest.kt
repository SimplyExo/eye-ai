package com.algorithmic_alliance.eyeaiapp.settingsparser

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs on an Android device because TensorFlow Lite's JNI runtime is Android
 * native code. The fixture is a fixed Python selected-pair export and is
 * packaged only in the test APK.
 */
@RunWith(AndroidJUnit4::class)
class CleanV2GoldenParityInstrumentationTest {
	@Test
	fun productionAndroidPipelineEqualsFixedPythonReferenceCommands() {
		val instrumentation = InstrumentationRegistry.getInstrumentation()
		val payload = instrumentation.context.assets
			.open("settingsparser/clean_v2_golden_commands.json")
			.bufferedReader()
			.use { JSONObject(it.readText()) }
		assertEquals(SettingsTfliteContract.ARCHITECTURE, payload.getString("architecture"))
		assertEquals(20260812, payload.getInt("word_seed"))
		assertEquals(20260814, payload.getInt("character_seed"))

		val parser = LocalSettingsParser.fromAssets(instrumentation.targetContext)
		try {
			val cases = payload.getJSONArray("cases")
			for (index in 0 until cases.length()) {
				val expected = cases.getJSONObject(index)
				val id = expected.getString("id")
				val command = parser.parse(
					SettingTarget.valueOf(expected.getString("target")),
					expected.getString("text")
				)
				assertEquals(id, SettingOperation.valueOf(expected.getString("operation")), command.operation)
				assertDoubleOrNull(id, expected, "numeric_value", command.numericValue)
				assertEquals(id, nullableEnum(expected, "magnitude", ChangeMagnitude::valueOf), command.magnitude)
				assertEquals(id, nullableEnum(expected, "speaker", SpeakerChoice::valueOf), command.speaker)
				assertEquals(id, nullableEnum(expected, "unit", SettingUnit::valueOf), command.unit)
				assertEquals(id, SettingParseStatus.valueOf(expected.getString("status")), command.status)
				assertEquals(id, expected.getString("normalized_text"), command.normalizedText)
				assertEquals(id, NumberNormalizationStatus.valueOf(expected.getString("number_status")), command.numberNormalizationStatus)
				assertEquals(id, doubleList(expected.getJSONArray("extracted_numeric_values")), command.extractedNumericValues)
				assertEquals(id, stringList(expected.getJSONArray("diagnostics")), command.diagnostics)
				assertEquals(id, Text2NumGermanNumberNormalizer.NORMALIZER_ID, command.normalizerId)
				assertEquals(id, Text2NumGermanNumberNormalizer.NORMALIZER_VERSION, command.normalizerVersion)
				if (expected.has("raw_number_occurrence_values")) {
					assertEquals(
						id,
						doubleList(expected.getJSONArray("raw_number_occurrence_values")),
						command.numberOccurrences.filter { it.status == NumberOccurrenceStatus.SUCCESS }.map { it.value!! }
					)
				}
			}
		} finally {
			parser.close()
		}
	}

	private fun assertDoubleOrNull(id: String, expected: JSONObject, key: String, actual: Double?) {
		if (expected.isNull(key)) {
			assertNull(id, actual)
		} else {
			assertEquals(id, expected.getDouble(key), requireNotNull(actual), 0.0)
		}
	}

	private fun <T> nullableEnum(
		item: JSONObject,
		key: String,
		factory: (String) -> T
	): T? = if (item.isNull(key)) null else factory(item.getString(key))

	private fun doubleList(values: JSONArray): List<Double> = List(values.length()) { values.getDouble(it) }

	private fun stringList(values: JSONArray): List<String> = List(values.length()) { values.getString(it) }
}
