package com.algorithmic_alliance.eyeaiapp.settingsparser

import android.content.res.AssetManager
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import org.json.JSONObject

/** Frozen, APK-safe Clean-v2 artifact identity. No training artifact is loaded at runtime. */
object SettingsParserAssetContract {
	const val ASSET_DIRECTORY = "nlp-v2/settings-parser"
	const val CONTRACT_ASSET = "$ASSET_DIRECTORY/settings_parser_contract.json"
	const val WORD_MODEL_ASSET = "$ASSET_DIRECTORY/word_operation_seed_20260812.tflite"
	const val CHARACTER_MODEL_ASSET = "$ASSET_DIRECTORY/character_speaker_seed_20260814.tflite"
	const val WORD_TOKENIZER_ASSET = "$ASSET_DIRECTORY/word_tokenizer.json"
	const val CHARACTER_TOKENIZER_ASSET = "$ASSET_DIRECTORY/character_tokenizer.json"

	const val WORD_MODEL_SHA256 = "0b992d94767c87629d4e1044d097638bcc2a85a9c4050ea3719e7c55009f0519"
	const val CHARACTER_MODEL_SHA256 = "fd61e69b450378cf91991c3900dd966fd412492ff9e5be10db82e231989b4a79"
	const val WORD_TOKENIZER_SHA256 = "6f87b77a9609b82c7bec09c4450d98b892a84549edd1086e8d03419c9da64405"
	const val CHARACTER_TOKENIZER_SHA256 = "6b7a7b71f686a07eb14e45c37bb99653d1855e5a26e2e8c41a5cdef5285067d0"

	data class VerifiedAssets(
		val wordTokenizerJson: String,
		val characterTokenizerJson: String
	)

	private data class ExpectedAsset(val path: String, val sha256: String)

	private val expectedAssets = listOf(
		ExpectedAsset(WORD_MODEL_ASSET, WORD_MODEL_SHA256),
		ExpectedAsset(CHARACTER_MODEL_ASSET, CHARACTER_MODEL_SHA256),
		ExpectedAsset(WORD_TOKENIZER_ASSET, WORD_TOKENIZER_SHA256),
		ExpectedAsset(CHARACTER_TOKENIZER_ASSET, CHARACTER_TOKENIZER_SHA256)
	)

	/** Verifies the exact package contents once before interpreter creation. */
	fun verifyAssets(assetManager: AssetManager): VerifiedAssets {
		val expectedFileNames = (expectedAssets.map { fileName(it.path) } + fileName(CONTRACT_ASSET)).sorted()
		val actualFileNames = assetManager.list(ASSET_DIRECTORY)?.sorted().orEmpty()
		require(actualFileNames == expectedFileNames) {
			"Settings-parser APK asset directory contains non-production files: $actualFileNames"
		}
		val content = expectedAssets.associate { expected ->
			val bytes = assetManager.open(expected.path).use { it.readBytes() }
			require(sha256(bytes) == expected.sha256) {
				"Frozen settings-parser asset SHA mismatch: ${expected.path}"
			}
			expected.path to bytes
		}
		val contract = assetManager.open(CONTRACT_ASSET).bufferedReader().use { it.readText() }
		verifyContract(contract)
		return VerifiedAssets(
			wordTokenizerJson = requireNotNull(content[WORD_TOKENIZER_ASSET]).toString(Charsets.UTF_8),
			characterTokenizerJson = requireNotNull(content[CHARACTER_TOKENIZER_ASSET]).toString(Charsets.UTF_8)
		)
	}

	/** Test/build equivalent that needs neither an Android Context nor an AssetManager. */
	fun verifyDirectory(directory: Path): VerifiedAssets {
		require(Files.isDirectory(directory)) { "Missing settings-parser asset directory: $directory" }
		val expectedFileNames = expectedAssets.map { fileName(it.path) } + fileName(CONTRACT_ASSET)
		Files.newDirectoryStream(directory).use { entries ->
			val actual = entries.map { it.fileName.toString() }.sorted()
			require(actual == expectedFileNames.sorted()) {
				"Settings-parser asset directory contains non-production files: $actual"
			}
		}
		for (expected in expectedAssets) {
			val file = directory.resolve(fileName(expected.path))
			require(sha256(Files.readAllBytes(file)) == expected.sha256) {
				"Frozen settings-parser asset SHA mismatch: $file"
			}
		}
		verifyContract(Files.readString(directory.resolve(fileName(CONTRACT_ASSET))))
		return VerifiedAssets(
			wordTokenizerJson = Files.readString(directory.resolve(fileName(WORD_TOKENIZER_ASSET))),
			characterTokenizerJson = Files.readString(directory.resolve(fileName(CHARACTER_TOKENIZER_ASSET)))
		)
	}

	private fun verifyContract(serialized: String) {
		val contract = JSONObject(serialized)
		require(contract.getInt("schema_version") == 1)
		require(contract.getString("architecture") == SettingsTfliteContract.ARCHITECTURE)
		val normalizer = contract.getJSONObject("normalizer")
		require(
			normalizer.getString("id") == SettingsTfliteContract.NORMALIZER_ID &&
				normalizer.getString("version") == SettingsTfliteContract.NORMALIZER_VERSION
		)
		verifyModelContract(
			contract.getJSONObject("word_operation"),
			asset = fileName(WORD_MODEL_ASSET),
			modelSha = WORD_MODEL_SHA256,
			tokenizer = fileName(WORD_TOKENIZER_ASSET),
			tokenizerSha = WORD_TOKENIZER_SHA256,
			maxLen = SettingsTfliteContract.WORD_MAX_LEN,
			activeOutput = SettingsTfliteContract.OPERATION_OUTPUT_NAME,
			outputSize = SettingOperation.entries.size,
			normalization = FrozenSettingsTokenizer.NORMALIZATION_SPEC_VERSION
		)
		verifyModelContract(
			contract.getJSONObject("character_speaker"),
			asset = fileName(CHARACTER_MODEL_ASSET),
			modelSha = CHARACTER_MODEL_SHA256,
			tokenizer = fileName(CHARACTER_TOKENIZER_ASSET),
			tokenizerSha = CHARACTER_TOKENIZER_SHA256,
			maxLen = SettingsTfliteContract.CHARACTER_MAX_LEN,
			activeOutput = SettingsTfliteContract.SPEAKER_OUTPUT_NAME,
			outputSize = SpeakerChoice.entries.size,
			normalization = FrozenCharacterSettingsTokenizer.NORMALIZATION_SPEC_VERSION
		)
		val routing = contract.getJSONObject("head_routing")
		require(routing.getString("operation") == "word_operation.active_output")
		require(routing.getString("speaker") == "character_speaker.active_output")
	}

	private fun verifyModelContract(
		contract: JSONObject,
		asset: String,
		modelSha: String,
		tokenizer: String,
		tokenizerSha: String,
		maxLen: Int,
		activeOutput: String,
		outputSize: Int,
		normalization: String
	) {
		require(contract.getString("asset") == asset)
		require(contract.getString("sha256") == modelSha)
		require(contract.getString("tokenizer_asset") == tokenizer)
		require(contract.getString("tokenizer_sha256") == tokenizerSha)
		require(contract.getString("normalization_spec_version") == normalization)
		val input = contract.getJSONObject("input")
		require(
			input.getString("name") == SettingsTfliteContract.INPUT_NAME &&
				input.getString("dtype") == SettingsTfliteContract.INPUT_DTYPE &&
				input.getJSONArray("shape").let { it.getInt(0) == 1 && it.getInt(1) == maxLen }
		)
		val output = contract.getJSONObject("active_output")
		require(
			output.getString("name") == activeOutput &&
				output.getString("dtype") == "float32" &&
				output.getJSONArray("shape").let { it.getInt(0) == 1 && it.getInt(1) == outputSize } &&
				output.getJSONArray("classes").length() == outputSize
		)
	}

	private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
		.digest(bytes)
		.joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

	private fun fileName(path: String): String = path.substringAfterLast('/')
}
