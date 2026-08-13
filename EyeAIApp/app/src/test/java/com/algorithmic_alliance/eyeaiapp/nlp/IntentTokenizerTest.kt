package com.algorithmic_alliance.eyeaiapp.nlp

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class IntentTokenizerTest {
	@Test
	fun normalizationMatchesTrainingPipeline() {
		assertEquals("öffne bitte", IntentTokenizer.normalize("  ÖFFNE, bitte!  "))
		assertEquals("nicht abbrechen", IntentTokenizer.normalize("Nicht – abbrechen"))
		assertEquals("eins zwei", IntentTokenizer.normalize("eins\u0085zwei"))
	}

	@Test
	fun wordTokenizerUsesUnknownIdAndPostPadding() {
		val tokenizer = IntentTokenizer(
			vocabulary = listOf("[PAD]", "[UNK]", "lies", "text"),
			maxLength = 5,
			type = IntentTokenizerType.WORD
		)

		assertArrayEquals(intArrayOf(2, 1, 3, 0, 0), tokenizer.encode("Lies den Text!"))
	}

	@Test
	fun bpeTokenizerAppliesMergesByRankAndPostTruncates() {
		val tokenizer = IntentTokenizer(
			vocabulary = listOf("[PAD]", "[UNK]", "▁", "a", "b", "▁a", "ab", "▁ab"),
			maxLength = 2,
			type = IntentTokenizerType.BPE,
			bpeMerges = listOf(
				BpeMerge(0, "▁", "a", "▁a"),
				BpeMerge(1, "a", "b", "ab"),
				BpeMerge(2, "▁a", "b", "▁ab")
			)
		)

		assertEquals(listOf("▁ab", "▁ab", "▁ab"), tokenizer.tokenize("ab ab ab"))
		assertArrayEquals(intArrayOf(7, 7), tokenizer.encode("ab ab ab"))
	}
}
