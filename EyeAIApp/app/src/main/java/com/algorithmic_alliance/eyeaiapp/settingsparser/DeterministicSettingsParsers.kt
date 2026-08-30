package com.algorithmic_alliance.eyeaiapp.settingsparser

enum class MagnitudeParseStatus {
	CLEAR,
	AMBIGUOUS
}

data class MagnitudeParseResult(
	val value: ChangeMagnitude?,
	val status: MagnitudeParseStatus,
	val diagnostic: String? = null
)

class GermanMagnitudeParser {
	private val smallPatterns = listOf(
		Regex("\\betwas\\b", RegexOption.IGNORE_CASE),
		Regex("\\bein\\s+bisschen\\b", RegexOption.IGNORE_CASE),
		Regex("\\bein\\s+wenig\\b", RegexOption.IGNORE_CASE),
		Regex("\\bleicht\\b", RegexOption.IGNORE_CASE),
		Regex("\\bminimal\\b", RegexOption.IGNORE_CASE),
		Regex("\\bgeringf(?:ü|ue)gig\\b", RegexOption.IGNORE_CASE)
	)
	private val largePatterns = listOf(
		Regex("\\bdeutlich\\b", RegexOption.IGNORE_CASE),
		Regex("\\bstark\\b", RegexOption.IGNORE_CASE),
		Regex("\\bviel\\b", RegexOption.IGNORE_CASE),
		Regex("\\bordentlich\\b", RegexOption.IGNORE_CASE),
		Regex("\\berheblich\\b", RegexOption.IGNORE_CASE)
	)
	private val negatedMarker = Regex(
		"\\b(?:nicht\\s+(?:so\\s+)?(?:etwas|ein\\s+bisschen|ein\\s+wenig|leicht|minimal|geringf(?:ü|ue)gig|deutlich|stark|viel|ordentlich|erheblich)|kein(?:e|en|em|er)?\\s+(?:bisschen|wenig))\\b",
		RegexOption.IGNORE_CASE
	)

	fun parse(
		text: String,
		operation: SettingOperation,
		numericValue: Double?
	): MagnitudeParseResult {
		if (operation !in setOf(SettingOperation.INCREASE, SettingOperation.DECREASE) || numericValue != null) {
			return MagnitudeParseResult(null, MagnitudeParseStatus.CLEAR)
		}
		val small = smallPatterns.any { it.containsMatchIn(text) }
		val large = largePatterns.any { it.containsMatchIn(text) }
		return when {
			negatedMarker.containsMatchIn(text) -> MagnitudeParseResult(
				null,
				MagnitudeParseStatus.AMBIGUOUS,
				"NEGATED_MAGNITUDE"
			)
			small && large -> MagnitudeParseResult(
				null,
				MagnitudeParseStatus.AMBIGUOUS,
				"CONFLICTING_MAGNITUDE_MARKERS"
			)
			small -> MagnitudeParseResult(ChangeMagnitude.SMALL, MagnitudeParseStatus.CLEAR)
			large -> MagnitudeParseResult(ChangeMagnitude.LARGE, MagnitudeParseStatus.CLEAR)
			else -> MagnitudeParseResult(ChangeMagnitude.DEFAULT, MagnitudeParseStatus.CLEAR)
		}
	}
}

data class UnitParseResult(
	val unit: SettingUnit?,
	val explicitUnit: SettingUnit?,
	val isValid: Boolean,
	val diagnostic: String? = null,
	val unsupportedUnit: String? = null
)

class SettingUnitParser {
	private val hertz = Regex("\\b(?:hertz|hz)\\b", RegexOption.IGNORE_CASE)
	private val bps = Regex(
		"\\b(?:b\\s*p\\s*s|beats?\\s+per\\s+second|(?:impuls|impulse|puls|pulse|schl(?:ä|ae)ge)\\s+pro\\s+sekunde)\\b",
		RegexOption.IGNORE_CASE
	)
	private val speechRate = Regex(
		"\\b(?:sprechgeschwindigkeit|sprachgeschwindigkeit|speech\\s*rate)\\b",
		RegexOption.IGNORE_CASE
	)
	private val unsupported = Regex("\\b(?:kiloherz|kilohertz|khz|prozent)\\b|%", RegexOption.IGNORE_CASE)

	fun parse(target: SettingTarget, text: String): UnitParseResult {
		unsupported.find(text)?.let {
			return UnitParseResult(null, null, false, "UNSUPPORTED_EXPLICIT_UNIT", it.value)
		}
		var foundBps = bps.containsMatchIn(text)
		val foundHertz = hertz.containsMatchIn(text)
		// "BPS auf ... Hertz" names the target and carries one explicit wrong
		// unit. The target mention must not turn this into a fake conflict.
		if (
			target == SettingTarget.BPS &&
			foundHertz &&
			foundBps &&
			Regex("\\bbps\\b", RegexOption.IGNORE_CASE).containsMatchIn(text)
		) {
			foundBps = false
		}
		val found = buildList {
			if (foundHertz) add(SettingUnit.HZ)
			if (foundBps) add(SettingUnit.BPS)
			if (speechRate.containsMatchIn(text)) add(SettingUnit.SPEECH_RATE)
		}.distinct()
		if (found.size > 1) {
			return UnitParseResult(null, null, false, "CONFLICTING_EXPLICIT_UNITS")
		}
		val explicit = found.firstOrNull()
		val expected = defaultUnit(target)
		if (explicit != null && explicit != expected) {
			return UnitParseResult(explicit, explicit, false, "INVALID_UNIT_FOR_${target.name}")
		}
		return UnitParseResult(explicit ?: expected, explicit, true)
	}

	companion object {
		fun defaultUnit(target: SettingTarget): SettingUnit? = when (target) {
			SettingTarget.FREQUENCY -> SettingUnit.HZ
			SettingTarget.BPS -> SettingUnit.BPS
			SettingTarget.SPEECH_SPEED -> SettingUnit.SPEECH_RATE
			SettingTarget.SPEAKER -> null
		}
	}
}
