package com.algorithmic_alliance.eyeaiapp.settingsparser

import java.util.Locale

/**
 * Dependency-free Android port of the pinned `text2num_rs_de` 3.0.2 adapter
 * used for Clean-v2. It intentionally preserves the same occurrence/mask
 * contract: signs remain lexical model input while only numeric text is masked.
 */
class Text2NumGermanNumberNormalizer : GermanNumberNormalizer {
	override fun normalize(text: String): NumberNormalizationResult {
		val matches = numberToken.findAll(text).map(::TokenMatch).toList()
		val occurrences = mutableListOf<NumberOccurrence>()
		var index = 0
		while (index < matches.size) {
			var current = matches[index]
			var token = current.raw
			var sign = 1.0
			var start = current.start
			var valueIndex = index

			val folded = fold(token)
			if (folded in signWords) {
				if (!signPrefixesNumber(matches, index, text)) {
					index++
					continue
				}
				sign = signWords.getValue(folded)
				valueIndex++
				current = matches[valueIndex]
				token = current.raw
			}

			if (digitToken.matches(token)) {
				val value = coerceNumber(token) * sign
				val maskStart = current.start + if (token.firstOrNull() in setOf('+', '-')) 1 else 0
				occurrences += successfulOccurrence(text, start, current.end, value, maskStart)
				index = valueIndex + 1
				continue
			}

			// text2num only reports connector fragments when they touch a
			// neighboring numeric parse. A bare "und" in a speaker utterance is
			// ordinary language; "eins komma" retains the incomplete comma as a
			// PARTIAL_FAILURE after the successful "eins" prefix.
			if (fold(token) in connectorWords) {
				if (connectorTouchesNumber(matches, valueIndex, occurrences, text)) {
					occurrences += NumberOccurrence(
						originalText = current.raw,
						value = null,
						start = current.start,
						end = current.end,
						status = NumberOccurrenceStatus.PARTIAL_FAILURE
					)
				}
				index++
				continue
			}

			if (!isNumberishWord(token) || !articleIsNumeric(matches, valueIndex, text)) {
				index++
				continue
			}

			var endIndex = valueIndex
			val words = mutableListOf<String>()
			while (endIndex < matches.size) {
				val candidate = matches[endIndex]
				if (endIndex > valueIndex && !isContiguous(text, matches[endIndex - 1].end, candidate.start)) {
					break
				}
				if (
					digitToken.matches(candidate.raw) ||
					!isNumberishWord(candidate.raw) ||
					!articleIsNumeric(matches, endIndex, text)
				) break
				words += candidate.raw
				endIndex++
			}

			val parsed = longestParseablePrefix(words)
			val parsedWordCount = parsed?.second ?: 0
			val value = parsed?.first
			if (value != null) {
				val end = matches[valueIndex + parsedWordCount - 1].end
				occurrences += successfulOccurrence(text, start, end, value * sign, matches[valueIndex].start)
				index = valueIndex + parsedWordCount
			} else {
				// Match text2num's token API behavior: keep an unparseable grammar
				// fragment as a failed occurrence, then continue scanning so a later
				// valid number is not swallowed by one malformed phrase.
				occurrences += NumberOccurrence(
					originalText = text.substring(start, current.end),
					value = null,
					start = start,
					end = current.end,
					status = NumberOccurrenceStatus.PARTIAL_FAILURE
				)
				index = valueIndex + 1
			}
		}

		val ordered = occurrences.sortedWith(compareBy<NumberOccurrence> { it.start == null }.thenBy { it.start ?: 0 })
		val values = ordered.filter { it.status == NumberOccurrenceStatus.SUCCESS }.map { requireNotNull(it.value) }
		return NumberNormalizationResult(
			originalText = text,
			normalizedText = maskedText(text, ordered),
			values = values,
			occurrences = ordered,
			status = overallStatus(ordered),
			normalizerId = NORMALIZER_ID,
			normalizerVersion = NORMALIZER_VERSION
		)
	}

	private fun successfulOccurrence(
		text: String,
		start: Int,
		end: Int,
		value: Double,
		maskStart: Int
	): NumberOccurrence = NumberOccurrence(
		originalText = text.substring(start, end),
		value = value,
		start = start,
		end = end,
		status = NumberOccurrenceStatus.SUCCESS,
		maskStart = maskStart,
		maskEnd = end
	)

	private fun maskedText(text: String, occurrences: List<NumberOccurrence>): String {
		val pieces = StringBuilder(text.length)
		var cursor = 0
		for (occurrence in occurrences) {
			val start = occurrence.start ?: continue
			val end = occurrence.end ?: continue
			val maskStart = occurrence.maskStart ?: start
			val maskEnd = occurrence.maskEnd ?: end
			if (maskStart < cursor) continue
			pieces.append(text, cursor, maskStart)
			pieces.append("<NUM>")
			cursor = maskEnd
		}
		pieces.append(text, cursor, text.length)
		return pieces.toString()
	}

	private fun overallStatus(occurrences: List<NumberOccurrence>): NumberNormalizationStatus {
		val failures = occurrences.filter { it.status != NumberOccurrenceStatus.SUCCESS }
		return when {
			failures.any { it.status == NumberOccurrenceStatus.INVALID } -> NumberNormalizationStatus.INVALID
			failures.isNotEmpty() -> NumberNormalizationStatus.PARTIAL_FAILURE
			occurrences.none { it.status == NumberOccurrenceStatus.SUCCESS } -> NumberNormalizationStatus.NO_NUMBER
			occurrences.count { it.status == NumberOccurrenceStatus.SUCCESS } == 1 -> NumberNormalizationStatus.SUCCESS
			else -> NumberNormalizationStatus.AMBIGUOUS
		}
	}

	private fun signPrefixesNumber(matches: List<TokenMatch>, index: Int, text: String): Boolean {
		if (fold(matches[index].raw) !in signWords || index + 1 >= matches.size) return false
		val following = matches[index + 1]
		if (!isContiguous(text, matches[index].end, following.start)) return false
		return digitToken.matches(following.raw) ||
			(isNumberishWord(following.raw) && articleIsNumeric(matches, index + 1, text))
	}

	private fun articleIsNumeric(matches: List<TokenMatch>, index: Int, text: String): Boolean {
		val folded = fold(matches[index].raw)
		if (folded !in articleForms) return true
		if (index + 1 >= matches.size) return false
		val following = matches[index + 1]
		if (!isContiguous(text, matches[index].end, following.start)) return false
		val next = fold(following.raw)
		return next in scaleOrDecimalContinuations || (folded == "ein" && next in explicitOneUnits)
	}

	private fun isContiguous(text: String, leftEnd: Int, rightStart: Int): Boolean =
		spacingOnly.matches(text.substring(leftEnd, rightStart))

	private fun isNumberishWord(word: String): Boolean {
		val folded = fold(word)
		return folded in numberishWords || parseCompound(folded) != null
	}

	private fun longestParseablePrefix(words: List<String>): Pair<Double, Int>? {
		for (size in words.size downTo 1) {
			parseWordPhrase(words.take(size))?.let { return it to size }
		}
		return null
	}

	private fun connectorTouchesNumber(
		matches: List<TokenMatch>,
		index: Int,
		occurrences: List<NumberOccurrence>,
		text: String
	): Boolean {
		val previousTokenClaimed = index > 0 && occurrences.any { occurrence ->
			occurrence.status == NumberOccurrenceStatus.SUCCESS &&
			occurrence.start != null && occurrence.end != null &&
			occurrence.start <= matches[index - 1].start && occurrence.end >= matches[index - 1].end
		}
		val nextIsNumberish = index + 1 < matches.size &&
			isNumberishWord(matches[index + 1].raw) &&
			articleIsNumeric(matches, index + 1, text)
		return previousTokenClaimed || nextIsNumberish
	}

	private fun parseWordPhrase(words: List<String>): Double? {
		val canonical = words.map(::fold)
		if (canonical.isEmpty() || canonical.count { it == "komma" } > 1) return null
		if ("komma" in canonical) {
			val comma = canonical.indexOf("komma")
			val integerWords = canonical.subList(0, comma)
			val decimalWords = canonical.subList(comma + 1, canonical.size)
			if (integerWords.isEmpty() || decimalWords.isEmpty()) return null
			val integer = parseCompound(integerWords.joinToString("")) ?: return null
			val decimalParts = decimalWords.map(::parseCompound)
			if (decimalParts.any { it == null }) return null
			val decimalDigits = decimalParts.joinToString("") { requireNotNull(it).toString() }
			return coerceNumber("$integer.$decimalDigits")
		}
		return parseCompound(canonical.joinToString(""))?.toDouble()
	}

	private fun parseCompound(input: String): Int? {
		val word = fold(input)
		direct[word]?.let { return it }
		for ((marker, multiplier) in scales) {
			val position = word.indexOf(marker)
			if (position >= 0) {
				val left = word.substring(0, position)
				val right = word.substring(position + marker.length)
				val leftValue = if (left.isEmpty()) 1 else parseCompound(left) ?: return null
				var result = leftValue * multiplier
				val additiveRight = right.removePrefix("und")
				if (additiveRight.isNotEmpty()) result += parseCompound(additiveRight) ?: return null
				return result
			}
		}
		val und = word.indexOf("und")
		if (und >= 0) {
			val left = parseCompound(word.substring(0, und))
			val right = tens[word.substring(und + "und".length)]
			if (left != null && right != null && left in 1..9) return left + right
		}
		return null
	}

	private fun coerceNumber(raw: String): Double = raw.replace(',', '.').toDouble()

	private fun fold(word: String): String = word.lowercase(Locale.ROOT)
		.replace("ä", "ae")
		.replace("ö", "oe")
		.replace("ü", "ue")
		.replace("ß", "ss")

	private data class TokenMatch(val raw: String, val start: Int, val end: Int) {
		constructor(match: MatchResult) : this(match.value, match.range.first, match.range.last + 1)
	}

	companion object {
		const val NORMALIZER_ID = SettingsTfliteContract.NORMALIZER_ID
		const val NORMALIZER_VERSION = SettingsTfliteContract.NORMALIZER_VERSION

		private val numberToken = Regex("[+-]?\\d+(?:[,.]\\d+)?|[A-Za-zÄÖÜäöüß]+")
		private val digitToken = Regex("^[+-]?\\d+(?:[,.]\\d+)?$")
		private val spacingOnly = Regex("^[\\s-]*$")
		private val signWords = mapOf("minus" to -1.0, "plus" to 1.0)
		private val connectorWords = setOf("komma", "und")
		private val direct = mapOf(
			"null" to 0, "ein" to 1, "eins" to 1, "eine" to 1, "einen" to 1,
			"einem" to 1, "einer" to 1, "zwei" to 2, "drei" to 3, "vier" to 4,
			"fuenf" to 5, "sechs" to 6, "sieben" to 7, "acht" to 8, "neun" to 9,
			"zehn" to 10, "elf" to 11, "zwoelf" to 12, "dreizehn" to 13,
			"vierzehn" to 14, "fuenfzehn" to 15, "sechzehn" to 16, "siebzehn" to 17,
			"achtzehn" to 18, "neunzehn" to 19, "zwanzig" to 20, "dreissig" to 30,
			"vierzig" to 40, "fuenfzig" to 50, "sechzig" to 60, "siebzig" to 70,
			"achtzig" to 80, "neunzig" to 90
		)
		private val tens = direct.filterValues { it in 20..90 && it % 10 == 0 }
		private val articleForms = setOf("ein", "eine", "einen", "einem", "einer")
		private val scaleOrDecimalContinuations = setOf(
			"hundert", "tausend", "million", "millionen", "milliarde", "milliarden", "komma"
		)
		private val explicitOneUnits = setOf("hertz", "hz", "bps")
		private val numberishWords = direct.keys + scaleOrDecimalContinuations + "und"
		private val scales = listOf(
			"milliarden" to 1_000_000_000,
			"milliarde" to 1_000_000_000,
			"millionen" to 1_000_000,
			"million" to 1_000_000,
			"tausend" to 1_000,
			"hundert" to 100
		)
	}
}
