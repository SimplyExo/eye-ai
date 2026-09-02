package com.algorithmic_alliance.eyeaiapp.settingsparser

import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import org.json.JSONObject

class SettingsParserContractsTest {
	@Test
	fun `existing intents map to targets without adding a new classifier`() {
		assertEquals(SettingTarget.FREQUENCY, SettingTarget.fromIntent(Intent.SET_FREQUENCY))
		assertEquals(SettingTarget.BPS, SettingTarget.fromIntent(Intent.SET_BPS))
		assertEquals(SettingTarget.SPEECH_SPEED, SettingTarget.fromIntent(Intent.CHANGE_SPEECH_SPEED))
		assertEquals(SettingTarget.SPEAKER, SettingTarget.fromIntent(Intent.CHANGE_SPEAKER))
		assertEquals(null, SettingTarget.fromIntent(Intent.OBJECT_DETECTION))
	}

	@Test
	fun `frozen tokenizer prepends stable context and masks unknown words`() {
		val vocabulary = FrozenSettingsTokenizer.SPECIAL_TOKENS.withIndex().associate { it.value to it.index } + mapOf("erhöhe" to 7)
		val tokenizer = FrozenSettingsTokenizer(vocabulary)
		val encoded = tokenizer.encodeWithContext(SettingTarget.FREQUENCY, "erhöhe <NUM> unbekannt")
		assertArrayEquals(intArrayOf(2, 7, 6, 1) + IntArray(28), encoded)
	}

	@Test
	fun `shared Python Kotlin tokenizer golden vectors are identical`() {
		val tokenizerJson = requireNotNull(javaClass.getResource("/tokenizer_golden_tokenizer.json")).readText()
		val vectorsJson = requireNotNull(javaClass.getResource("/tokenizer_golden_vectors.json")).readText()
		val tokenizer = FrozenSettingsTokenizer.fromJson(tokenizerJson)
		val payload = JSONObject(vectorsJson)
		assertEquals(
			FrozenSettingsTokenizer.NORMALIZATION_SPEC_VERSION,
			payload.getString("normalization_spec_version")
		)
		val vectors = payload.getJSONArray("vectors")
		for (index in 0 until vectors.length()) {
			val vector = vectors.getJSONObject(index)
			val input = vector.getString("input")
			assertEquals(
				vector.getString("expected_normalized_text"),
				FrozenSettingsTokenizer.normalizeTokenizerText(input)
			)
			val expected = vector.getJSONArray("expected_token_ids")
			assertArrayEquals(
				IntArray(expected.length()) { expected.getInt(it) },
				tokenizer.encodeWithContext(SettingTarget.valueOf(vector.getString("target")), input)
			)
		}
	}
}
