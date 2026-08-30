package com.algorithmic_alliance.eyeaiapp.settingsparser

import java.text.Normalizer
import java.util.Locale
import org.json.JSONObject

/** Exact text and target-context preparation shared by the two frozen encoders. */
private object SettingsTokenizerTextContract {
	val wordSpecialTokens = listOf(
		"[PAD]", "[UNK]", "[CTX_FREQ]", "[CTX_BPS]", "[CTX_SPEED]", "[CTX_SPEAKER]", "<NUM>"
	)
	private val tokenPattern = Regex("<NUM>|\\[[A-Z_]+]|[A-Za-zÄÖÜäöüß]+|\\d+(?:[,.]\\d+)?")
	private val asciiWhitespace = Regex("[\\u0009-\\u000D\\u0020]+")

	fun contextToken(target: SettingTarget): String = when (target) {
		SettingTarget.FREQUENCY -> "[CTX_FREQ]"
		SettingTarget.BPS -> "[CTX_BPS]"
		SettingTarget.SPEECH_SPEED -> "[CTX_SPEED]"
		SettingTarget.SPEAKER -> "[CTX_SPEAKER]"
	}

	/** Python training rule: NFC and ASCII whitespace canonicalization only. */
	fun normalize(text: String): String = asciiWhitespace.replace(
		Normalizer.normalize(text, Normalizer.Form.NFC),
		" "
	).trim()

	fun tokenizedWords(text: String): List<String> = tokenPattern.findAll(normalize(text)).map { match ->
		val token = match.value
		if (token in wordSpecialTokens) token else token.lowercase(Locale.ROOT)
	}.toList()

	fun contextualText(target: SettingTarget, normalizedText: String): String =
		"${contextToken(target)} $normalizedText"
}

/**
 * Frozen word-tokenizer contract from Clean-v2. The context token is prepended
 * before tokenization, and post-truncation/post-padding produces int32[1,32].
 */
class FrozenSettingsTokenizer(
	private val vocabulary: Map<String, Int>,
	val maxLen: Int = SettingsTfliteContract.WORD_MAX_LEN
) {
	init {
		require(maxLen == SettingsTfliteContract.WORD_MAX_LEN) {
			"Word tokenizer length must be ${SettingsTfliteContract.WORD_MAX_LEN}"
		}
		require(vocabulary.values.sorted() == (0 until vocabulary.size).toList()) {
			"Word tokenizer ids must be contiguous and zero based"
		}
		require(SPECIAL_TOKENS.withIndex().all { (index, token) -> vocabulary[token] == index }) {
			"Tokenizer special-token ids do not match the frozen Settings contract"
		}
	}

	fun encodeWithContext(target: SettingTarget, normalizedText: String): IntArray =
		encode(SettingsTokenizerTextContract.contextualText(target, normalizedText))

	fun encode(text: String): IntArray {
		val ids = IntArray(maxLen) { vocabulary.getValue(PAD) }
		for ((index, token) in tokenize(text).take(maxLen).withIndex()) {
			ids[index] = vocabulary[token] ?: vocabulary.getValue(UNK)
		}
		return ids
	}

	internal fun tokenize(text: String): List<String> = SettingsTokenizerTextContract.tokenizedWords(text)

	companion object {
		const val TOKENIZER_SCHEMA_VERSION = 2
		const val NORMALIZATION_SPEC_VERSION = "eyeai_word_v1"
		const val PAD = "[PAD]"
		const val UNK = "[UNK]"
		const val CTX_FREQ = "[CTX_FREQ]"
		const val CTX_BPS = "[CTX_BPS]"
		const val CTX_SPEED = "[CTX_SPEED]"
		const val CTX_SPEAKER = "[CTX_SPEAKER]"
		const val NUM = "<NUM>"
		val SPECIAL_TOKENS = SettingsTokenizerTextContract.wordSpecialTokens

		fun contextTokenForTarget(target: SettingTarget): String =
			SettingsTokenizerTextContract.contextToken(target)

		fun normalizeTokenizerText(text: String): String = SettingsTokenizerTextContract.normalize(text)

		fun fromJson(json: String): FrozenSettingsTokenizer {
			val payload = JSONObject(json)
			require(payload.getInt("schema_version") == TOKENIZER_SCHEMA_VERSION)
			require(payload.getString("kind") == "word")
			require(payload.getString("normalization_spec_version") == NORMALIZATION_SPEC_VERSION)
			val special = payload.getJSONArray("special_tokens")
			require((0 until special.length()).map(special::getString) == SPECIAL_TOKENS)
			val rawVocabulary = payload.getJSONObject("vocabulary")
			val vocabulary = rawVocabulary.keys().asSequence().associateWith(rawVocabulary::getInt)
			return FrozenSettingsTokenizer(vocabulary, payload.getInt("max_len"))
		}
	}
}

/**
 * Frozen Character-CNN tokenizer. It deliberately encodes every character of
 * the lowercased contextual text and therefore produces int32[1,96].
 */
class FrozenCharacterSettingsTokenizer(
	private val vocabulary: Map<String, Int>,
	val maxLen: Int = SettingsTfliteContract.CHARACTER_MAX_LEN
) {
	init {
		require(maxLen == SettingsTfliteContract.CHARACTER_MAX_LEN) {
			"Character tokenizer length must be ${SettingsTfliteContract.CHARACTER_MAX_LEN}"
		}
		require(vocabulary.values.sorted() == (0 until vocabulary.size).toList()) {
			"Character tokenizer ids must be contiguous and zero based"
		}
		require(vocabulary[PAD] == 0 && vocabulary[UNK] == 1) {
			"Character tokenizer special-token ids must be PAD=0 and UNK=1"
		}
	}

	fun encodeWithContext(target: SettingTarget, normalizedText: String): IntArray =
		encode(contextualText(target, normalizedText))

	fun encode(text: String): IntArray {
		val ids = IntArray(maxLen) { vocabulary.getValue(PAD) }
		text.codePoints().iterator().asSequence().take(maxLen).forEachIndexed { index, codePoint ->
			ids[index] = vocabulary[String(Character.toChars(codePoint))] ?: vocabulary.getValue(UNK)
		}
		return ids
	}

	fun contextualText(target: SettingTarget, normalizedText: String): String =
		SettingsTokenizerTextContract.normalize(
			SettingsTokenizerTextContract.contextualText(target, normalizedText)
		).lowercase(Locale.ROOT)

	companion object {
		const val TOKENIZER_SCHEMA_VERSION = 1
		const val NORMALIZATION_SPEC_VERSION = "eyeai_char_v1"
		const val PAD = "[PAD]"
		const val UNK = "[UNK]"

		fun fromJson(json: String): FrozenCharacterSettingsTokenizer {
			val payload = JSONObject(json)
			require(payload.getInt("schema_version") == TOKENIZER_SCHEMA_VERSION)
			require(payload.getString("kind") == "character")
			require(payload.getString("normalization_spec_version") == NORMALIZATION_SPEC_VERSION)
			val special = payload.getJSONArray("special_tokens")
			require((0 until special.length()).map(special::getString) == listOf(PAD, UNK))
			val rawVocabulary = payload.getJSONObject("vocabulary")
			val vocabulary = rawVocabulary.keys().asSequence().associateWith(rawVocabulary::getInt)
			return FrozenCharacterSettingsTokenizer(vocabulary, payload.getInt("max_len"))
		}
	}
}
