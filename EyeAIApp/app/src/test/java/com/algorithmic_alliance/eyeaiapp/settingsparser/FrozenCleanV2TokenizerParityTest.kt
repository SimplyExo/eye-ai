package com.algorithmic_alliance.eyeaiapp.settingsparser

import java.nio.file.Files
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/** Python reference vectors generated with the frozen Clean-v2 tokenizer artifacts. */
class FrozenCleanV2TokenizerParityTest {
	private val assetDirectory = settingsParserAssetDirectory()

	@Test
	fun `word tokenizer has Python parity including target context and padding`() {
		val tokenizer = FrozenSettingsTokenizer.fromJson(
			Files.readString(assetDirectory.resolve("word_tokenizer.json"))
		)
		val cases = listOf(
			Triple(SettingTarget.FREQUENCY, "erhöhe <NUM> hertz", intArrayOf(2, 49, 6, 13)),
			Triple(SettingTarget.BPS, "minus <NUM> bps", intArrayOf(3, 90, 6, 8)),
			Triple(SettingTarget.SPEECH_SPEED, "schneller", intArrayOf(4, 48)),
			Triple(SettingTarget.SPEAKER, "männliche stimme", intArrayOf(5, 52, 14))
		)
		for ((target, text, prefix) in cases) {
			assertArrayEquals(
				"$target: $text",
				prefix + IntArray(SettingsTfliteContract.WORD_MAX_LEN - prefix.size),
				tokenizer.encodeWithContext(target, text)
			)
		}
	}

	@Test
	fun `character tokenizer has Python parity including literal context and num marker`() {
		val tokenizer = FrozenCharacterSettingsTokenizer.fromJson(
			Files.readString(assetDirectory.resolve("character_tokenizer.json"))
		)
		val cases = listOf(
			Triple(
				SettingTarget.FREQUENCY,
				"erhöhe <NUM> hertz",
				intArrayOf(5, 10, 27, 31, 7, 13, 25, 12, 24, 6, 2, 12, 25, 15, 34, 15, 12, 2, 3, 21, 28, 20, 4, 2, 15, 12, 25, 27, 32)
			),
			Triple(
				SettingTarget.BPS,
				"minus <NUM> bps",
				intArrayOf(5, 10, 27, 31, 7, 9, 23, 26, 6, 2, 20, 16, 21, 28, 26, 2, 3, 21, 28, 20, 4, 2, 9, 23, 26)
			),
			Triple(
				SettingTarget.SPEECH_SPEED,
				"schneller",
				intArrayOf(5, 10, 27, 31, 7, 26, 23, 12, 12, 11, 6, 2, 26, 10, 15, 21, 12, 19, 19, 12, 25)
			),
			Triple(
				SettingTarget.SPEAKER,
				"männliche stimme",
				intArrayOf(5, 10, 27, 31, 7, 26, 23, 12, 8, 18, 12, 25, 6, 2, 20, 33, 21, 21, 19, 16, 10, 15, 12, 2, 26, 27, 16, 20, 20, 12)
			)
		)
		for ((target, text, prefix) in cases) {
			assertArrayEquals(
				"$target: $text",
				prefix + IntArray(SettingsTfliteContract.CHARACTER_MAX_LEN - prefix.size),
				tokenizer.encodeWithContext(target, text)
			)
		}
		assertEquals("[ctx_speaker] männliche stimme", tokenizer.contextualText(SettingTarget.SPEAKER, "männliche stimme"))
	}
}
