package com.algorithmic_alliance.eyeaiapp.settingsparser

import org.junit.Assert.assertEquals
import org.junit.Test

/** Fixed vectors from Python's pinned text2num==3.0.2 adapter. */
class Text2NumGermanNumberNormalizerTest {
	private val normalizer = Text2NumGermanNumberNormalizer()

	@Test
	fun `pinned normalizer preserves lexical signs and source ranges`() {
		val result = normalizer.normalize("minus einhundert")
		assertEquals("minus <NUM>", result.normalizedText)
		assertEquals(listOf(-100.0), result.values)
		assertEquals(NumberNormalizationStatus.SUCCESS, result.status)
		assertEquals(Text2NumGermanNumberNormalizer.NORMALIZER_ID, result.normalizerId)
		assertEquals(Text2NumGermanNumberNormalizer.NORMALIZER_VERSION, result.normalizerVersion)
		val occurrence = result.occurrences.single()
		assertEquals(-100.0, occurrence.value)
		assertEquals(0, occurrence.start)
		assertEquals(16, occurrence.end)
		assertEquals(6, occurrence.maskStart)
		assertEquals(16, occurrence.maskEnd)
	}

	@Test
	fun `pinned normalizer matches Python decimal article and connector vectors`() {
		val cases = listOf(
			Golden("eins komma zwei", "<NUM>", listOf(1.2), NumberNormalizationStatus.SUCCESS),
			Golden("für die bps stell bitte fünf komma acht ein", "für die bps stell bitte <NUM> ein", listOf(5.8), NumberNormalizationStatus.SUCCESS),
			Golden("stell b p s fünf ein", "stell b p s <NUM> ein", listOf(5.0), NumberNormalizationStatus.SUCCESS),
			Golden("wechsel die stimme und nimm die maskuline", "wechsel die stimme und nimm die maskuline", emptyList(), NumberNormalizationStatus.NO_NUMBER),
			Golden("eins komma", "<NUM> <NUM>", listOf(1.0), NumberNormalizationStatus.PARTIAL_FAILURE),
			Golden("von sechshundert auf siebenhundert", "von <NUM> auf <NUM>", listOf(600.0, 700.0), NumberNormalizationStatus.AMBIGUOUS),
			Golden("ein bisschen schneller", "ein bisschen schneller", emptyList(), NumberNormalizationStatus.NO_NUMBER),
			Golden("minus -100", "minus -<NUM>", listOf(100.0), NumberNormalizationStatus.SUCCESS)
		)
		for (case in cases) {
			val actual = normalizer.normalize(case.text)
			assertEquals(case.text, case.normalized, actual.normalizedText)
			assertEquals(case.text, case.values, actual.values)
			assertEquals(case.text, case.status, actual.status)
		}
	}

	private data class Golden(
		val text: String,
		val normalized: String,
		val values: List<Double>,
		val status: NumberNormalizationStatus
	)
}
