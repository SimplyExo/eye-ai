package com.algorithmic_alliance.eyeaiapp.settingsparser

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SpecializedSettingsTfliteRuntimeTest {
	@Test
	fun `asset contract verifies exact frozen product files`() {
		val verified = SettingsParserAssetContract.verifyDirectory(settingsParserAssetDirectory())
		assertTrue(verified.wordTokenizerJson.contains("\"kind\": \"word\""))
		assertTrue(verified.characterTokenizerJson.contains("\"kind\": \"character\""))
	}

	@Test
	fun `head routing is word to operation and character to speaker only`() {
		val word = RecordingWordHead(floatArrayOf(0.01f, 0.90f, 0.02f, 0.03f, 0.04f))
		val character = RecordingCharacterHead(floatArrayOf(0.04f, 0.91f, 0.05f))
		val predictors = SpecializedSettingsTflitePredictors(
			wordTokenizer = FrozenSettingsTokenizer(
				FrozenSettingsTokenizer.SPECIAL_TOKENS.withIndex().associate { it.value to it.index }
			),
			characterTokenizer = FrozenCharacterSettingsTokenizer(mapOf("[PAD]" to 0, "[UNK]" to 1)),
			wordOperationHead = word,
			characterSpeakerHead = character
		)

		val operation = predictors.predictOperation(SettingTarget.FREQUENCY, "mehr")
		val speaker = predictors.predictSpeaker(SettingTarget.SPEAKER, "feminin")

		assertEquals(SettingOperation.INCREASE, operation.operation)
		assertEquals(SpeakerChoice.FEMALE, speaker.speaker)
		assertEquals(1, word.calls)
		assertEquals(1, character.calls)
		assertEquals(SettingsTfliteContract.WORD_MAX_LEN, word.lastInput!!.size)
		assertEquals(SettingsTfliteContract.CHARACTER_MAX_LEN, character.lastInput!!.size)
		assertArrayEquals(
			floatArrayOf(0.01f, 0.90f, 0.02f, 0.03f, 0.04f),
			operation.probabilities,
			0f
		)
	}

	private class RecordingWordHead(private val answer: FloatArray) : WordOperationHead {
		var calls = 0
		var lastInput: IntArray? = null
		override fun inferOperation(tokenIds: IntArray): FloatArray {
			calls++
			lastInput = tokenIds.copyOf()
			return answer.copyOf()
		}
	}

	private class RecordingCharacterHead(private val answer: FloatArray) : CharacterSpeakerHead {
		var calls = 0
		var lastInput: IntArray? = null
		override fun inferSpeaker(tokenIds: IntArray): FloatArray {
			calls++
			lastInput = tokenIds.copyOf()
			return answer.copyOf()
		}
	}
}
