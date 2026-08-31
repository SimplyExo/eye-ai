package com.algorithmic_alliance.eyeaiapp.nlp

import java.text.Normalizer
import java.util.Locale

enum class IntentTokenizerType(val serializedName: String) {
	WORD("deterministic_word_level"),
	BPE("deterministic_word_boundary_bpe");

	companion object {
		fun fromSerializedName(value: String): IntentTokenizerType =
			entries.firstOrNull { it.serializedName == value }
				?: throw IllegalArgumentException("Unsupported tokenizer type: $value")
	}
}

data class BpeMerge(
	val rank: Int,
	val left: String,
	val right: String,
	val merged: String
)

/**
 * Deterministic tokenizer used by the NLP V2 training pipeline.
 *
 * Both padding and truncation happen at the end of the sequence. Token ID 0 is
 * reserved for padding and token ID 1 for unknown input. Its rules are loaded
 * from and validated against the frozen tokenizer artifacts; they must not be
 * changed independently from the training pipeline.
 */
class IntentTokenizer(
	vocabulary: List<String>,
	val maxLength: Int,
	val type: IntentTokenizerType,
	bpeMerges: List<BpeMerge> = emptyList()
) {
	private val vocabulary = vocabulary.toList()
	private val tokenToId = this.vocabulary.withIndex().associate { it.value to it.index }
	private val unknownToken = this.vocabulary.getOrNull(1)
		?: throw IllegalArgumentException("Vocabulary must contain an unknown token")
	private val mergesByPair: Map<Pair<String, String>, BpeMerge>

	init {
		require(maxLength > 0) { "Tokenizer maxLength must be positive" }
		require(this.vocabulary.firstOrNull() == PAD_TOKEN) {
			"Vocabulary ID $PAD_TOKEN_ID must be $PAD_TOKEN"
		}
		require(unknownToken == UNKNOWN_TOKEN) {
			"Vocabulary ID $UNKNOWN_TOKEN_ID must be $UNKNOWN_TOKEN"
		}
		require(this.vocabulary.size == tokenToId.size) {
			"Vocabulary must not contain duplicate tokens"
		}

		if (type == IntentTokenizerType.BPE) {
			require(bpeMerges.map { it.rank } == bpeMerges.indices.toList()) {
				"BPE merge ranks must be contiguous and ordered"
			}
			require(bpeMerges.all { it.merged in tokenToId }) {
				"Every BPE merge result must exist in the vocabulary"
			}
			mergesByPair = bpeMerges.associateBy { it.left to it.right }
			require(mergesByPair.size == bpeMerges.size) {
				"BPE merge pairs must be unique"
			}
		} else {
			require(bpeMerges.isEmpty()) { "Word tokenizers must not contain BPE merges" }
			mergesByPair = emptyMap()
		}
	}

	fun tokenize(text: String): List<String> {
		val normalized = normalize(text)
		if (normalized.isEmpty()) return emptyList()

		val words = normalized.split(' ')
		return if (type == IntentTokenizerType.BPE) {
			words.flatMap(::tokenizeBpeWord)
		} else {
			words
		}
	}

	fun encode(text: String): IntArray {
		val encoded = IntArray(maxLength)
		tokenize(text).take(maxLength).forEachIndexed { index, token ->
			encoded[index] = tokenToId[token] ?: UNKNOWN_TOKEN_ID
		}
		return encoded
	}

	private fun tokenizeBpeWord(word: String): List<String> {
		var symbols = buildList {
			add(WORD_BOUNDARY_SYMBOL)
			word.codePoints().forEach { codePoint ->
				val character = String(Character.toChars(codePoint))
				add(if (character in tokenToId) character else unknownToken)
			}
		}

		while (symbols.size > 1) {
			val selectedMerge = symbols.zipWithNext()
				.mapNotNull { mergesByPair[it] }
				.minByOrNull { it.rank }
				?: break

			val mergedSymbols = ArrayList<String>(symbols.size)
			var index = 0
			while (index < symbols.size) {
				if (
					index + 1 < symbols.size &&
					symbols[index] == selectedMerge.left &&
					symbols[index + 1] == selectedMerge.right
				) {
					mergedSymbols.add(selectedMerge.merged)
					index += 2
				} else {
					mergedSymbols.add(symbols[index])
					index++
				}
			}
			symbols = mergedSymbols
		}

		return symbols
	}

	companion object {
		const val NORMALIZATION_ID = "shared_intent_nfkc_lower_punctuation_whitespace_v1"
		const val PAD_TOKEN = "[PAD]"
		const val PAD_TOKEN_ID = 0
		const val UNKNOWN_TOKEN = "[UNK]"
		const val UNKNOWN_TOKEN_ID = 1
		const val WORD_BOUNDARY_SYMBOL = "▁"

		private val punctuationTypes = setOf(
			Character.CONNECTOR_PUNCTUATION.toInt(),
			Character.DASH_PUNCTUATION.toInt(),
			Character.START_PUNCTUATION.toInt(),
			Character.END_PUNCTUATION.toInt(),
			Character.INITIAL_QUOTE_PUNCTUATION.toInt(),
			Character.FINAL_QUOTE_PUNCTUATION.toInt(),
			Character.OTHER_PUNCTUATION.toInt()
		)

		/** Exact Android port of [NORMALIZATION_ID] from the training pipeline. */
		fun normalize(text: String): String {
			val normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
				.lowercase(Locale.ROOT)
			val characters = StringBuilder(normalized.length)
			normalized.codePoints().forEach { codePoint ->
				if (
					isPythonWhitespace(codePoint) ||
					Character.getType(codePoint) in punctuationTypes
				) {
					characters.append(' ')
				} else {
					characters.appendCodePoint(codePoint)
				}
			}
			return characters.toString()
				.trim()
				.split(Regex(" +"))
				.filter { it.isNotEmpty() }
				.joinToString(" ")
		}

		private fun isPythonWhitespace(codePoint: Int): Boolean =
			Character.isWhitespace(codePoint) ||
				Character.isSpaceChar(codePoint) ||
				codePoint == 0x0085
	}
}
