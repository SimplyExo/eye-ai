package com.algorithmic_alliance.eyeaiapp.confirmation

import android.content.Context
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.InputStream
import java.text.Normalizer
import java.util.Locale
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

enum class ConfirmationLabel {
	ACCEPT,
	REJECT,
	UNKNOWN
}

data class ConfirmationResult(
	val label: ConfirmationLabel,
	val confidence: Double,
	val scores: Map<ConfirmationLabel, Double>,
	val source: String,
	val decisionReason: String,
	val rawLabel: ConfirmationLabel = label,
	val confidenceThreshold: Double
) {
	fun toDecisionTraceFields(): String {
		val serializedScores = ConfirmationLabel.entries.joinToString(
			prefix = "[",
			postfix = "]"
		) { label -> "$label=${formatProbability(scores.getValue(label))}" }
		return "model=${ConfirmationModel.MODEL_ID} decision=$label " +
			"confirmed=${label == ConfirmationLabel.ACCEPT} " +
			"rejected=${label == ConfirmationLabel.REJECT} " +
			"requiresClarification=${label == ConfirmationLabel.UNKNOWN} " +
			"rawLabel=$rawLabel confidence=${formatProbability(confidence)} " +
			"threshold=${formatProbability(confidenceThreshold)} " +
			"confidenceRejected=${decisionReason == "low_char_confidence"} " +
			"source=$source reason=$decisionReason scores=$serializedScores"
	}

	private fun formatProbability(value: Double): String =
		String.format(Locale.US, "%.4f", value)
}

/**
 * Local confirmation pipeline exported from the prototype:
 * deterministic exact phrases -> character TF-IDF/logistic regression -> confidence UNKNOWN.
 */
class ConfirmationModel private constructor(
	private val minNgram: Int,
	private val maxNgram: Int,
	private val confidenceUnknownThreshold: Double,
	private val labels: List<ConfirmationLabel>,
	private val intercepts: DoubleArray,
	private val vocabulary: Map<String, Int>,
	private val idf: DoubleArray,
	private val coefficients: Array<DoubleArray>
) {
	val featureCount: Int get() = idf.size
	val confidenceThreshold: Double get() = confidenceUnknownThreshold

	fun classify(question: String, answer: String, pendingAction: String): ConfirmationResult {
		require(question.isNotBlank()) { "Confirmation question must not be blank" }
		require(answer.isNotBlank()) { "Confirmation answer must not be blank" }
		require(pendingAction.isNotBlank()) { "Pending confirmation action must not be blank" }

		fastRule(question, answer)?.let { return it }

		val rendered = "[FRAGE] $question [AKTION] $pendingAction [ANTWORT] $answer"
		val scores = predictProbabilities(rendered)
		val rawLabel = labels.maxBy { scores.getValue(it) }
		val confidence = scores.getValue(rawLabel)
		val rejectedForConfidence =
			rawLabel != ConfirmationLabel.UNKNOWN && confidence < confidenceUnknownThreshold

		return ConfirmationResult(
			label = if (rejectedForConfidence) ConfirmationLabel.UNKNOWN else rawLabel,
			confidence = confidence,
			scores = scores,
			source = if (rejectedForConfidence) {
				"char_ngram_confidence_reject"
			} else {
				"char_ngram"
			},
			decisionReason = if (rejectedForConfidence) {
				"low_char_confidence"
			} else {
				"semantic_prediction"
			},
			rawLabel = rawLabel,
			confidenceThreshold = confidenceUnknownThreshold
		)
	}

	private fun predictProbabilities(text: String): Map<ConfirmationLabel, Double> {
		val counts = HashMap<Int, Int>()
		for (ngram in characterWordBoundaryNgrams(text.lowercase(Locale.ROOT))) {
			val featureIndex = vocabulary[ngram] ?: continue
			counts[featureIndex] = (counts[featureIndex] ?: 0) + 1
		}

		val weightedFeatures = HashMap<Int, Double>(counts.size)
		var squaredNorm = 0.0
		for ((featureIndex, count) in counts) {
			val value = (1.0 + ln(count.toDouble())) * idf[featureIndex]
			weightedFeatures[featureIndex] = value
			squaredNorm += value * value
		}
		val norm = sqrt(squaredNorm)

		val logits = intercepts.copyOf()
		if (norm > 0.0) {
			for ((featureIndex, value) in weightedFeatures) {
				val normalizedValue = value / norm
				for (classIndex in labels.indices) {
					logits[classIndex] +=
						coefficients[classIndex][featureIndex] * normalizedValue
				}
			}
		}

		val maximum = logits.max()
		val exponentials = DoubleArray(logits.size) { index -> exp(logits[index] - maximum) }
		val sum = exponentials.sum()
		return labels.indices.associate { index -> labels[index] to exponentials[index] / sum }
	}

	private fun characterWordBoundaryNgrams(text: String): Sequence<String> = sequence {
		for (word in splitLikePythonWhitespace(text)) {
			val padded = " $word "
			val wordLength = padded.length
			for (size in minNgram..maxNgram) {
				var offset = 0
				yield(padded.substring(offset, minOf(offset + size, wordLength)))
				while (offset + size < wordLength) {
					offset++
					yield(padded.substring(offset, offset + size))
				}
				if (offset == 0) break
			}
		}
	}

	private fun fastRule(question: String, answer: String): ConfirmationResult? {
		val normalizedAnswer = normalizeRuleText(answer)
		val label = normalizedRules.entries.firstOrNull { normalizedAnswer in it.value }?.key
			?: return null
		if (label != ConfirmationLabel.UNKNOWN && questionIsNegated(question)) return null

		return ConfirmationResult(
			label = label,
			confidence = 1.0,
			scores = ConfirmationLabel.entries.associateWith { if (it == label) 1.0 else 0.0 },
			source = "fast_rule",
			decisionReason = "deterministic_exact_phrase",
			confidenceThreshold = confidenceUnknownThreshold
		)
	}

	private fun questionIsNegated(question: String): Boolean {
		val normalized = " ${normalizeRuleText(question)} "
		return listOf(" nicht ", " kein ", " keine ").any(normalized::contains)
	}

	companion object {
		const val MODEL_ID = "deterministic_char_ngram_v1"
		const val ASSET_PATH = "confirmation/char_ngram_v1.bin"
		private const val MAGIC = "EYEAI_CONFIRMATION_CHAR_NGRAM\n"
		private const val FORMAT_VERSION = 1
		private const val MAX_STRING_BYTES = 1_000_000
		private val expectedLabels = ConfirmationLabel.entries.toList()

		private val rules = mapOf(
			ConfirmationLabel.ACCEPT to setOf(
				"ja", "ja bitte", "ja genau", "jo", "joa", "jep", "jap", "jup",
				"klar", "gerne", "okay", "ok", "mach", "mach das", "bitte",
				"von mir aus", "warum nicht", "kannst machen", "ja kannst du machen",
				"jau", "jawohl", "passt", "tu das", "meinetwegen", "ja gern",
				"genau", "stimmt", "absolut", "auf jeden fall", "natürlich",
				"einverstanden", "klar doch", "geht klar", "klingt gut", "richtig",
				"mach weiter", "sehr gern", "sehr gerne", "selbstverständlich"
			),
			ConfirmationLabel.REJECT to setOf(
				"nein", "nee", "ne", "nö", "nein danke", "lieber nicht",
				"nicht nötig", "auf keinen fall", "brauch ich nicht", "lass es",
				"nein lieber nicht", "ähm nein", "bitte nicht", "keinesfalls",
				"lass mal", "bloß nicht", "muss nicht sein", "besser nicht", "stop",
				"stopp", "abbrechen", "niemals", "negativ", "absolut nicht",
				"auf gar keinen fall", "kommt nicht in frage", "vergiss es",
				"leider nicht", "kein interesse"
			),
			ConfirmationLabel.UNKNOWN to setOf(
				"weiß nicht", "weiß ich nicht", "keine ahnung", "vielleicht",
				"mal schauen", "moment", "wie meinst du das",
				"kann ich gerade nicht sagen", "äh keine ahnung",
				"unentschieden", "schwierig", "unklar", "eventuell", "noch offen",
				"kann sein", "hm", "möglicherweise", "kommt drauf an",
				"es kommt drauf an", "ich bin unsicher", "keine entscheidung",
				"was genau", "später", "noch nicht", "schwer zu sagen",
				"kannst du das erklären"
			)
		)
		private val normalizedRules = rules.mapValues { (_, phrases) ->
			phrases.mapTo(hashSetOf(), ::normalizeRuleText)
		}

		fun fromAssets(context: Context): ConfirmationModel =
			context.assets.open(ASSET_PATH).use(::load)

		fun load(input: InputStream): ConfirmationModel {
			DataInputStream(BufferedInputStream(input)).use { data ->
				val magic = ByteArray(MAGIC.toByteArray(Charsets.UTF_8).size)
				data.readFully(magic)
				require(magic.contentEquals(MAGIC.toByteArray(Charsets.UTF_8))) {
					"Invalid confirmation model header"
				}
				require(data.readInt() == FORMAT_VERSION) {
					"Unsupported confirmation model format"
				}
				val minNgram = data.readInt()
				val maxNgram = data.readInt()
				require(minNgram > 0 && maxNgram >= minNgram) { "Invalid n-gram range" }
				val threshold = data.readDouble()
				require(threshold in 0.0..1.0) { "Invalid confirmation threshold" }

				val labelCount = data.readInt()
				require(labelCount == expectedLabels.size) { "Unexpected label count" }
				val labels = List(labelCount) { ConfirmationLabel.valueOf(data.readString()) }
				require(labels == expectedLabels) { "Unexpected confirmation label order" }
				val intercepts = DoubleArray(labelCount) { data.readDouble() }

				val featureCount = data.readInt()
				require(featureCount in 1..100_000) { "Invalid feature count" }
				val vocabulary = HashMap<String, Int>(featureCount)
				val idf = DoubleArray(featureCount)
				val coefficients = Array(labelCount) { DoubleArray(featureCount) }
				for (featureIndex in 0 until featureCount) {
					val token = data.readString()
					require(vocabulary.put(token, featureIndex) == null) {
						"Duplicate confirmation feature"
					}
					idf[featureIndex] = data.readDouble()
					for (classIndex in 0 until labelCount) {
						coefficients[classIndex][featureIndex] = data.readDouble()
					}
				}
				require(data.read() == -1) { "Trailing bytes in confirmation model" }

				return ConfirmationModel(
					minNgram = minNgram,
					maxNgram = maxNgram,
					confidenceUnknownThreshold = threshold,
					labels = labels,
					intercepts = intercepts,
					vocabulary = vocabulary,
					idf = idf,
					coefficients = coefficients
				)
			}
		}

		private fun DataInputStream.readString(): String {
			val size = readInt()
			require(size in 0..MAX_STRING_BYTES) { "Invalid string size in confirmation model" }
			val bytes = ByteArray(size)
			readFully(bytes)
			return bytes.toString(Charsets.UTF_8)
		}

		private fun splitLikePythonWhitespace(text: String): List<String> {
			val words = mutableListOf<String>()
			val current = StringBuilder()
			text.codePoints().forEach { codePoint ->
				if (isPythonWhitespace(codePoint)) {
					if (current.isNotEmpty()) {
						words.add(current.toString())
						current.setLength(0)
					}
				} else {
					current.appendCodePoint(codePoint)
				}
			}
			if (current.isNotEmpty()) words.add(current.toString())
			return words
		}

		private fun normalizeRuleText(text: String): String {
			val normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
				.lowercase(Locale.ROOT)
				.replace("ß", "ss")
			val output = StringBuilder(normalized.length)
			var previousWasSpace = true
			normalized.codePoints().forEach { codePoint ->
				val keep = Character.isLetterOrDigit(codePoint) || codePoint == '_'.code
				if (keep) {
					output.appendCodePoint(codePoint)
					previousWasSpace = false
				} else if (!previousWasSpace) {
					output.append(' ')
					previousWasSpace = true
				}
			}
			return output.toString().trim()
		}

		private fun isPythonWhitespace(codePoint: Int): Boolean =
			Character.isWhitespace(codePoint) ||
				Character.isSpaceChar(codePoint) ||
				codePoint == 0x0085
	}
}
