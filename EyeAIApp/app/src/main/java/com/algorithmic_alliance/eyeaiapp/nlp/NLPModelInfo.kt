package com.algorithmic_alliance.eyeaiapp.nlp

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

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
