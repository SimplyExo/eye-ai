package com.algorithmic_alliance.eyeaiapp.nlp

import org.junit.Assert.assertArrayEquals
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path

/**
 * Frozen Python-training vectors for the real T1/T2 artifacts shipped in the app.
 * A mismatch here means Android would feed different int32 values to LiteRT.
 */
class FrozenTokenizerParityTest {
	@Test
	fun androidEncodingMatchesFrozenPythonVectors() {
		val t1 = loadTokenizer("T1")
		val t2 = loadTokenizer("T2")

		PARITY_CASES.forEach { parityCase ->
			assertArrayEquals("T1: ${parityCase.text}", parityCase.t1, t1.encode(parityCase.text))
			assertArrayEquals("T2: ${parityCase.text}", parityCase.t2, t2.encode(parityCase.text))
		}
	}

	private fun loadTokenizer(family: String): IntentTokenizer {
		val directory = tokenizerAssets().resolve(family)
		val vocabulary = parseStringArray(directory.resolve("vocab.json"))
		val merges = if (family == "T2") parseMerges(directory.resolve("merges.json")) else emptyList()
		return IntentTokenizer(
			vocabulary = vocabulary,
			maxLength = NLPModel.INPUT_LENGTH,
			type = if (family == "T2") IntentTokenizerType.BPE else IntentTokenizerType.WORD,
			bpeMerges = merges
		)
	}

	private fun tokenizerAssets(): Path {
		val candidates = listOf(
			Path.of("src/main/assets/nlp-v2/tokenizers"),
			Path.of("app/src/main/assets/nlp-v2/tokenizers")
		)
		return candidates.firstOrNull(Files::isDirectory)
			?: error("Cannot locate frozen NLP V2 tokenizer assets from ${Path.of("").toAbsolutePath()}")
	}

	private fun parseStringArray(path: Path): List<String> =
		Files.readAllLines(path)
			.map(String::trim)
			.filter { it.startsWith('"') }
			.map { serialized ->
				require('\\' !in serialized) { "Test parser does not accept escaped tokens" }
				serialized.removeSuffix(",").removeSurrounding("\"")
			}

	private fun parseMerges(path: Path): List<BpeMerge> {
		val serialized = Files.readString(path)
		val pattern = Regex(
			"""\{\s*"rank":\s*(\d+),\s*"left":\s*"([^"]*)",\s*"right":\s*"([^"]*)",\s*"merged":\s*"([^"]*)"\s*}"""
		)
		return pattern.findAll(serialized).map { match ->
			BpeMerge(
				rank = match.groupValues[1].toInt(),
				left = match.groupValues[2],
				right = match.groupValues[3],
				merged = match.groupValues[4]
			)
		}.toList()
	}

	private data class ParityCase(
		val text: String,
		val t1: IntArray,
		val t2: IntArray
	)

	companion object {
		private val PARITY_CASES = listOf(
			ParityCase(
				text = "  ÖFFNE, die Einstellungen!  ",
				t1 = intArrayOf(142, 2, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
				t2 = intArrayOf(524, 53, 234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
			),
			ParityCase(
				text = "Stelle die Frequenz auf 800 Hertz",
				t1 = intArrayOf(419, 2, 34, 12, 973, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
				t2 = intArrayOf(1094, 53, 208, 96, 1014, 385, 391, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
			),
			ParityCase(
				text = "Erkläre mir die Relativitätstheorie",
				t1 = intArrayOf(1052, 11, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
				t2 = intArrayOf(365, 87, 103, 53, 227, 63, 1870, 709, 1262, 47, 20, 17, 101, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
			),
			ParityCase(
				text = "Wie weit ist die Tür entfernt",
				t1 = intArrayOf(10, 25, 4, 2, 46, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
				t2 = intArrayOf(86, 153, 85, 53, 259, 414, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
			),
			ParityCase(
				text = "Brich den Vorgang ab",
				t1 = intArrayOf(462, 9, 424, 217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
				t2 = intArrayOf(1191, 98, 1100, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
			)
		)
	}
}
