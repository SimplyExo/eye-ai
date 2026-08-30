package com.algorithmic_alliance.eyeaiapp.tts

import java.text.Normalizer
import java.util.Locale

/** The small part of Android's Voice metadata needed for deterministic selection. */
internal data class TtsVoiceDescriptor(
	val name: String,
	val locale: Locale,
	val features: Set<String> = emptySet(),
	val isNetworkConnectionRequired: Boolean = false,
	val quality: Int = 0,
	val latency: Int = Int.MAX_VALUE
)

internal data class TtsVoiceSelection(
	val voice: TtsVoiceDescriptor?,
	val requestedSpeakerAvailable: Boolean
)

/**
 * Selects only voices that are in the backend's current catalog. Android does
 * not expose a standard gender field. Explicit gender metadata therefore
 * wins, but when the backend exposes at least two installed German voices we
 * choose two different voice families/variants using stable catalog metadata.
 * This keeps the abstract FEMALE/MALE setting useful without inventing a Voice
 * ID; every selected voice still comes directly from the backend catalog.
 */
internal object TtsVoiceSelector {
	const val FEMALE = 0
	const val MALE = 1

	fun select(
		requestedSpeaker: Int,
		voices: Collection<TtsVoiceDescriptor>,
		currentVoiceName: String? = null,
		defaultVoiceName: String? = null
	): TtsVoiceSelection {
		val usableVoices = voices
			.filter { it.name.isNotBlank() && isInstalled(it) }
			.distinctBy { it.name }
		val germanVoices = usableVoices
			.filter(::isGerman)
			.sortedWith(voiceComparator)
		val requestedVoice = speakerSlots(germanVoices)[requestedSpeaker]

		if (requestedVoice != null) {
			return TtsVoiceSelection(requestedVoice, requestedSpeakerAvailable = requestedSpeaker in 0..1)
		}

		return TtsVoiceSelection(
			voice = safeFallback(usableVoices, currentVoiceName, defaultVoiceName),
			requestedSpeakerAvailable = false
		)
	}

	private fun speakerSlots(
		germanVoices: List<TtsVoiceDescriptor>
	): Map<Int, TtsVoiceDescriptor?> {
		if (germanVoices.isEmpty()) return emptyMap()

		val explicitFemale = germanVoices.firstOrNull { matchesSpeaker(it, FEMALE) }
		val explicitMale = germanVoices.firstOrNull { matchesSpeaker(it, MALE) }

		if (explicitFemale == null && explicitMale == null) {
			val pair = mostDistinctPair(germanVoices)
			return mapOf(
				FEMALE to pair?.first,
				MALE to pair?.second
			)
		}

		val female = explicitFemale ?: explicitMale?.let {
			mostDistinctPartner(it, germanVoices)
		}
		val male = explicitMale ?: explicitFemale?.let {
			mostDistinctPartner(it, germanVoices)
		}

		if (female?.name == male?.name) {
			val alternativeMale = female?.let { mostDistinctPartner(it, germanVoices) }
			return mapOf(FEMALE to female, MALE to alternativeMale)
		}

		return mapOf(FEMALE to female, MALE to male)
	}

	private fun mostDistinctPair(
		voices: List<TtsVoiceDescriptor>
	): Pair<TtsVoiceDescriptor, TtsVoiceDescriptor>? {
		var bestPair: Pair<TtsVoiceDescriptor, TtsVoiceDescriptor>? = null
		var bestScore: VoicePairScore? = null

		for (firstIndex in 0 until voices.lastIndex) {
			for (secondIndex in firstIndex + 1 until voices.size) {
				val first = voices[firstIndex]
				val second = voices[secondIndex]
				val score = pairScore(first, second)
				if (bestScore == null || comparePairScores(score, bestScore) > 0) {
					bestPair = first to second
					bestScore = score
				}
			}
		}

		return bestPair
	}

	private fun mostDistinctPartner(
		reference: TtsVoiceDescriptor,
		voices: List<TtsVoiceDescriptor>
	): TtsVoiceDescriptor? {
		var bestVoice: TtsVoiceDescriptor? = null
		var bestScore: VoicePairScore? = null

		voices.filter { it.name != reference.name }.forEach { candidate ->
			val score = pairScore(reference, candidate)
			if (bestScore == null || comparePairScores(score, bestScore) > 0) {
				bestVoice = candidate
				bestScore = score
			}
		}

		return bestVoice
	}

	private data class VoicePairScore(
		val differentFamily: Int,
		val tokenDifference: Int,
		val localVoiceCount: Int,
		val quality: Int,
		val names: String
	)

	private fun pairScore(
		first: TtsVoiceDescriptor,
		second: TtsVoiceDescriptor
	): VoicePairScore {
		val firstFamily = familyKey(first)
		val secondFamily = familyKey(second)
		val firstTokens = voiceTokens(first)
		val secondTokens = voiceTokens(second)

		return VoicePairScore(
			differentFamily = if (
				firstFamily.isNotBlank() &&
				secondFamily.isNotBlank() &&
				firstFamily != secondFamily
			) 1 else 0,
			tokenDifference = (firstTokens - secondTokens).size +
				(secondTokens - firstTokens).size,
			localVoiceCount = listOf(first, second).count {
				!it.isNetworkConnectionRequired
			},
			quality = first.quality + second.quality,
			names = listOf(first.name, second.name).sorted().joinToString("|")
		)
	}

	private fun comparePairScores(
		left: VoicePairScore,
		right: VoicePairScore?
	): Int {
		if (right == null) return 1
		compareValues(left.differentFamily, right.differentFamily).takeIf { it != 0 }?.let { return it }
		compareValues(left.tokenDifference, right.tokenDifference).takeIf { it != 0 }?.let { return it }
		compareValues(left.localVoiceCount, right.localVoiceCount).takeIf { it != 0 }?.let { return it }
		compareValues(left.quality, right.quality).takeIf { it != 0 }?.let { return it }

		// Keep selection deterministic when the backend reports equivalent metadata.
		return compareValues(right.names, left.names)
	}

	private fun familyKey(voice: TtsVoiceDescriptor): String =
		voiceTokens(voice)
			.filterNot { it in GENERIC_VOICE_TOKENS || it.all(Char::isDigit) }
			.joinToString("|")

	private fun voiceTokens(voice: TtsVoiceDescriptor): Set<String> =
		(listOf(voice.name) + voice.features)
			.flatMap(::tokens)
			.toSet()

	private val GENERIC_VOICE_TOKENS = setOf(
		"de",
		"female",
		"fem",
		"frau",
		"herr",
		"language",
		"local",
		"male",
		"man",
		"mann",
		"masc",
		"network",
		"networktts",
		"online",
		"offline",
		"embedded",
		"embeddedtts",
		"default",
		"standard",
		"tts",
		"voice",
		"weib",
		"woman",
		"x"
	)

	private fun safeFallback(
		voices: List<TtsVoiceDescriptor>,
		currentVoiceName: String?,
		defaultVoiceName: String?
	): TtsVoiceDescriptor? {
		val germanVoices = voices.filter {
			it.locale.language.equals(Locale.GERMAN.language, ignoreCase = true)
		}
		val preferredPool = germanVoices.ifEmpty { voices }
		return preferredPool.firstOrNull { it.name == currentVoiceName }
			?: preferredPool.firstOrNull { it.name == defaultVoiceName }
			?: preferredPool.minWithOrNull(voiceComparator)
	}

	private fun isInstalled(voice: TtsVoiceDescriptor): Boolean =
		voice.features.none { feature ->
			normalize(feature).replace(" ", "").contains("notinstalled")
		}

	private fun isGerman(voice: TtsVoiceDescriptor): Boolean =
		voice.locale.language.equals(Locale.GERMAN.language, ignoreCase = true)

	private fun matchesSpeaker(voice: TtsVoiceDescriptor, requestedSpeaker: Int): Boolean {
		val tokens = (listOf(voice.name) + voice.features)
			.flatMap(::tokens)
		val female = tokens.any(::isFemaleToken)
		val male = tokens.any(::isMaleToken)
		return when (requestedSpeaker) {
			FEMALE -> female && !male
			MALE -> male && !female
			else -> false
		}
	}

	private fun isFemaleToken(token: String): Boolean =
		token.startsWith("female") ||
			token.startsWith("fem") ||
			token.startsWith("weib") ||
			token.startsWith("frau") ||
			token.startsWith("woman") ||
			token.startsWith("dame")

	private fun isMaleToken(token: String): Boolean =
		token.startsWith("male") ||
			token.startsWith("masc") ||
			token.startsWith("mann") ||
			token.startsWith("man") ||
			token.startsWith("herr")

	private fun tokens(value: String): List<String> =
		normalize(value).split(Regex("[^a-z0-9]+"))
			.filter(String::isNotBlank)

	private fun normalize(value: String): String =
		Normalizer.normalize(value, Normalizer.Form.NFKD)
			.replace(Regex("\\p{M}+"), "")
			.lowercase(Locale.ROOT)

	private val voiceComparator = compareByDescending<TtsVoiceDescriptor> { !it.isNetworkConnectionRequired }
		.thenByDescending { it.quality }
		.thenBy { it.latency }
		.thenBy { it.name }
}
