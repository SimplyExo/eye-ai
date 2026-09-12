package com.algorithmic_alliance.eyeaiapp.settingsparser

import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class SettingsParserContractsTest {
	@Test
	fun `existing intents map to targets without adding a new classifier`() {
		assertEquals(SettingTarget.FREQUENCY, SettingTarget.fromIntent(Intent.SET_FREQUENCY))
		assertEquals(SettingTarget.BPS, SettingTarget.fromIntent(Intent.SET_BPS))
		assertEquals(
			SettingTarget.SPEECH_SPEED, SettingTarget.fromIntent(Intent.CHANGE_SPEECH_SPEED)
		)
		assertEquals(SettingTarget.SPEAKER, SettingTarget.fromIntent(Intent.CHANGE_SPEAKER))
		assertEquals(null, SettingTarget.fromIntent(Intent.OBJECT_DETECTION))
	}

	@Test
	fun `frozen tokenizer prepends stable context and masks unknown words`() {
		val vocabulary = FrozenSettingsTokenizer.SPECIAL_TOKENS.withIndex()
			.associate { it.value to it.index } + mapOf("erhöhe" to 7)
		val tokenizer = FrozenSettingsTokenizer(vocabulary)
		val encoded = tokenizer.encodeWithContext(SettingTarget.FREQUENCY, "erhöhe <NUM> unbekannt")
		assertArrayEquals(intArrayOf(2, 7, 6, 1) + IntArray(28), encoded)
	}

}
