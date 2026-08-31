package com.algorithmic_alliance.eyeaiapp.nlp

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest

data class LoadedIntentTokenizer(
	val tokenizer: IntentTokenizer,
	val labels: List<String>
)

data class NLPModelInfo(
	val id: String,
	val displayName: String,
	val tfliteAssetPath: String,
	val tokenizerAssetDirectory: String
) {
	fun loadModelFile(context: Context): MappedByteBuffer {
		context.assets.openFd(tfliteAssetPath).use { fileDescriptor ->
			FileInputStream(fileDescriptor.fileDescriptor).channel.use { channel ->
				return channel.map(
					FileChannel.MapMode.READ_ONLY,
					fileDescriptor.startOffset,
					fileDescriptor.declaredLength
				)
			}
		}
	}

	fun loadTokenizer(context: Context): LoadedIntentTokenizer {
		val vocabulary = readJsonArray(context, "vocab.json")
		val labels = readJsonArray(context, "labels.json")
		val config = readJsonObject(context, "tokenizer_config.json")
		val tokenizerType = IntentTokenizerType.fromSerializedName(
			config.getString("tokenizer_type")
		)
		val merges = if (tokenizerType == IntentTokenizerType.BPE) {
			val serializedMerges = readJsonArrayObject(context, "merges.json")
			serializedMerges.map { merge ->
				BpeMerge(
					rank = merge.getInt("rank"),
					left = merge.getString("left"),
					right = merge.getString("right"),
					merged = merge.getString("merged")
				)
			}
		} else {
			emptyList()
		}
		validateTokenizerArtifacts(config, tokenizerType, vocabulary, merges)

		return LoadedIntentTokenizer(
			tokenizer = IntentTokenizer(
				vocabulary = vocabulary,
				maxLength = config.getInt("max_length"),
				type = tokenizerType,
				bpeMerges = merges
			),
			labels = labels
		)
	}

	private fun validateTokenizerArtifacts(
		config: JSONObject,
		tokenizerType: IntentTokenizerType,
		vocabulary: List<String>,
		merges: List<BpeMerge>
	) {
		require(config.getInt("version") == TOKENIZER_ARTIFACT_VERSION) {
			"Unsupported NLP V2 tokenizer artifact version"
		}
		require(config.getString("normalization") == IntentTokenizer.NORMALIZATION_ID) {
			"Unexpected NLP V2 normalization contract"
		}
		require(config.getString("padding") == "post") {
			"NLP V2 tokenizer must use post-padding"
		}
		require(config.getString("truncating") == "post") {
			"NLP V2 tokenizer must use post-truncation"
		}
		require(config.getInt("max_length") == NLPModel.INPUT_LENGTH) {
			"NLP V2 tokenizer max_length must be ${NLPModel.INPUT_LENGTH}"
		}

		val reservedTokens = config.getJSONObject("reserved_tokens")
		val padding = reservedTokens.getJSONObject("PAD")
		val unknown = reservedTokens.getJSONObject("UNK")
		require(
			padding.getString("token") == IntentTokenizer.PAD_TOKEN &&
				padding.getInt("id") == IntentTokenizer.PAD_TOKEN_ID
		) { "Unexpected NLP V2 padding token contract" }
		require(
			unknown.getString("token") == IntentTokenizer.UNKNOWN_TOKEN &&
				unknown.getInt("id") == IntentTokenizer.UNKNOWN_TOKEN_ID
		) { "Unexpected NLP V2 unknown token contract" }

		val expectedVocabularySize = when (tokenizerType) {
			IntentTokenizerType.WORD -> {
				require(config.getString("split") == "normalized whitespace") {
					"Unexpected NLP V2 word-tokenizer split rule"
				}
				require(merges.isEmpty()) { "Word tokenizer must not contain BPE merges" }
				config.getInt("vocabulary_size")
			}

			IntentTokenizerType.BPE -> {
				require(
					config.getString("word_boundary_symbol") == IntentTokenizer.WORD_BOUNDARY_SYMBOL
				) { "Unexpected NLP V2 BPE word-boundary symbol" }
				require(merges.size == config.getInt("merge_count")) {
					"NLP V2 BPE merge count does not match its artifact"
				}
				config.getInt("actual_vocabulary_size")
			}
		}
		require(vocabulary.size == expectedVocabularySize) {
			"NLP V2 vocabulary size ${vocabulary.size} does not match $expectedVocabularySize"
		}
		require(vocabularyChecksum(vocabulary) == config.getString("vocabulary_checksum_sha256")) {
			"NLP V2 vocabulary checksum does not match its frozen artifact"
		}
	}

	private fun vocabularyChecksum(vocabulary: List<String>): String {
		// The training exporter hashes the compact, UTF-8 JSON representation.
		val canonicalJson = JSONArray(vocabulary).toString()
		val digest = MessageDigest.getInstance("SHA-256")
			.digest(canonicalJson.toByteArray(Charsets.UTF_8))
		return buildString(digest.size * 2) {
			digest.forEach { byte ->
				val value = byte.toInt() and 0xff
				append(HEX_DIGITS[value ushr 4])
				append(HEX_DIGITS[value and 0x0f])
			}
		}
	}

	private fun readJsonObject(context: Context, filename: String): JSONObject =
		JSONObject(readAssetText(context, filename))

	private fun readJsonArray(context: Context, filename: String): List<String> {
		val array = JSONArray(readAssetText(context, filename))
		return List(array.length()) { index -> array.getString(index) }
	}

	private fun readJsonArrayObject(context: Context, filename: String): List<JSONObject> {
		val array = JSONArray(readAssetText(context, filename))
		return List(array.length()) { index -> array.getJSONObject(index) }
	}

	private fun readAssetText(context: Context, filename: String): String =
		context.assets.open("$tokenizerAssetDirectory/$filename")
			.bufferedReader()
			.use { it.readText() }

	companion object {
		private const val TOKENIZER_ARTIFACT_VERSION = 1
		private const val HEX_DIGITS = "0123456789abcdef"

		const val DEFAULT_MODEL_ID = "M0_T1"

		val BASELINE_MODELS = listOf(
			NLPModelInfo(
				id = "M0_T1",
				displayName = "M0 – Clean only – T1 Word – Seed 20260812",
				tfliteAssetPath = "nlp-v2/models/m0_t1_seed_20260812.tflite",
				tokenizerAssetDirectory = "nlp-v2/tokenizers/T1"
			),
			NLPModelInfo(
				id = "M0_T2",
				displayName = "M0 – Clean only – T2 BPE – Seed 20260814",
				tfliteAssetPath = "nlp-v2/models/m0_t2_seed_20260814.tflite",
				tokenizerAssetDirectory = "nlp-v2/tokenizers/T2"
			),
			NLPModelInfo(
				id = "M1_T1",
				displayName = "M1 – Joint Clean + Vosk – T1 Word – Seed 20260814",
				tfliteAssetPath = "nlp-v2/models/m1_t1_seed_20260814.tflite",
				tokenizerAssetDirectory = "nlp-v2/tokenizers/T1"
			),
			NLPModelInfo(
				id = "M1_T2",
				displayName = "M1 – Joint Clean + Vosk – T2 BPE – Seed 20260812",
				tfliteAssetPath = "nlp-v2/models/m1_t2_seed_20260812.tflite",
				tokenizerAssetDirectory = "nlp-v2/tokenizers/T2"
			),
			NLPModelInfo(
				id = "M2_T1",
				displayName = "M2 – Clean → Joint – T1 Word – Seed 20260813",
				tfliteAssetPath = "nlp-v2/models/m2_t1_seed_20260813.tflite",
				tokenizerAssetDirectory = "nlp-v2/tokenizers/T1"
			),
			NLPModelInfo(
				id = "M2_T2",
				displayName = "M2 – Clean → Joint – T2 BPE – Seed 20260814",
				tfliteAssetPath = "nlp-v2/models/m2_t2_seed_20260814.tflite",
				tokenizerAssetDirectory = "nlp-v2/tokenizers/T2"
			),
			NLPModelInfo(
				id = "M3_T1",
				displayName = "M3 – Clean → Vosk only – T1 Word – Seed 20260813",
				tfliteAssetPath = "nlp-v2/models/m3_t1_seed_20260813.tflite",
				tokenizerAssetDirectory = "nlp-v2/tokenizers/T1"
			),
			NLPModelInfo(
				id = "M3_T2",
				displayName = "M3 – Clean → Vosk only – T2 BPE – Seed 20260810",
				tfliteAssetPath = "nlp-v2/models/m3_t2_seed_20260810.tflite",
				tokenizerAssetDirectory = "nlp-v2/tokenizers/T2"
			)
		)

		fun findById(id: String): NLPModelInfo =
			BASELINE_MODELS.firstOrNull { it.id == id }
				?: BASELINE_MODELS.first { it.id == DEFAULT_MODEL_ID }
	}
}
