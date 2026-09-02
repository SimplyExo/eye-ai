package com.algorithmic_alliance.eyeaiapp.settingsparser

import android.content.Context
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

/** Narrow interfaces make the frozen head-routing independently testable. */
interface WordOperationHead {
	fun inferOperation(tokenIds: IntArray): FloatArray
}

interface CharacterSpeakerHead {
	fun inferSpeaker(tokenIds: IntArray): FloatArray
}

/**
 * Routes exactly one head from each model. Word-speaker and character-operation
 * probabilities are never requested from a production interpreter.
 */
class SpecializedSettingsTflitePredictors(
	private val wordTokenizer: FrozenSettingsTokenizer,
	private val characterTokenizer: FrozenCharacterSettingsTokenizer,
	private val wordOperationHead: WordOperationHead,
	private val characterSpeakerHead: CharacterSpeakerHead
) : OperationPredictor, SpeakerPredictor {
	override fun predictOperation(target: SettingTarget, normalizedText: String): OperationPrediction {
		val probabilities = wordOperationHead.inferOperation(
			wordTokenizer.encodeWithContext(target, normalizedText)
		)
		return operationPrediction(probabilities)
	}

	override fun predictSpeaker(target: SettingTarget, normalizedText: String): SpeakerPrediction {
		require(target == SettingTarget.SPEAKER) {
			"The Character-CNN speaker head is only valid for the SPEAKER target"
		}
		val probabilities = characterSpeakerHead.inferSpeaker(
			characterTokenizer.encodeWithContext(target, normalizedText)
		)
		return speakerPrediction(probabilities)
	}

	private fun operationPrediction(probabilities: FloatArray): OperationPrediction {
		requireProbabilityVector(probabilities, SettingOperation.entries.size, "word operation")
		val index = probabilities.indices.maxBy { probabilities[it] }
		return OperationPrediction(
			operation = SettingOperation.entries[index],
			confidence = probabilities[index],
			probabilities = probabilities.copyOf()
		)
	}

	private fun speakerPrediction(probabilities: FloatArray): SpeakerPrediction {
		requireProbabilityVector(probabilities, SpeakerChoice.entries.size, "character speaker")
		val index = probabilities.indices.maxBy { probabilities[it] }
		return SpeakerPrediction(
			speaker = SpeakerChoice.entries[index],
			confidence = probabilities[index],
			probabilities = probabilities.copyOf()
		)
	}

	private fun requireProbabilityVector(values: FloatArray, expectedSize: Int, source: String) {
		require(values.size == expectedSize) { "Unexpected $source vector size: ${values.size}" }
		require(values.all { it.isFinite() && it in 0f..1f }) { "Invalid $source probabilities" }
	}
}

/**
 * Long-lived, synchronized pair of TFLite interpreters. Create it once from
 * application assets and close it with the application lifecycle.
 */
class SpecializedSettingsTfliteRuntime private constructor(
	private val wordHead: TfliteWordOperationHead,
	private val characterHead: TfliteCharacterSpeakerHead,
	wordTokenizer: FrozenSettingsTokenizer,
	characterTokenizer: FrozenCharacterSettingsTokenizer
) : OperationPredictor, SpeakerPredictor, AutoCloseable {
	private val predictors = SpecializedSettingsTflitePredictors(
		wordTokenizer,
		characterTokenizer,
		wordHead,
		characterHead
	)
	private var closed = false

	@Synchronized
	override fun predictOperation(target: SettingTarget, normalizedText: String): OperationPrediction {
		check(!closed) { "Settings parser runtime is closed" }
		return predictors.predictOperation(target, normalizedText)
	}

	@Synchronized
	override fun predictSpeaker(target: SettingTarget, normalizedText: String): SpeakerPrediction {
		check(!closed) { "Settings parser runtime is closed" }
		return predictors.predictSpeaker(target, normalizedText)
	}

	@Synchronized
	override fun close() {
		if (closed) return
		closed = true
		wordHead.close()
		characterHead.close()
	}

	companion object {
		fun fromAssets(context: Context): SpecializedSettingsTfliteRuntime {
			val verified = SettingsParserAssetContract.verifyAssets(context.assets)
			return create(
				wordBuffer = mapAsset(context, SettingsParserAssetContract.WORD_MODEL_ASSET),
				characterBuffer = mapAsset(context, SettingsParserAssetContract.CHARACTER_MODEL_ASSET),
				wordTokenizer = FrozenSettingsTokenizer.fromJson(verified.wordTokenizerJson),
				characterTokenizer = FrozenCharacterSettingsTokenizer.fromJson(verified.characterTokenizerJson)
			)
		}

		/** Test helper for a TFLite-capable host using the identical APK asset files. */
		fun fromDirectory(directory: Path): SpecializedSettingsTfliteRuntime {
			val verified = SettingsParserAssetContract.verifyDirectory(directory)
			return create(
				wordBuffer = mapFile(directory.resolve("word_operation_seed_20260812.tflite")),
				characterBuffer = mapFile(directory.resolve("character_speaker_seed_20260814.tflite")),
				wordTokenizer = FrozenSettingsTokenizer.fromJson(verified.wordTokenizerJson),
				characterTokenizer = FrozenCharacterSettingsTokenizer.fromJson(verified.characterTokenizerJson)
			)
		}

		private fun create(
			wordBuffer: MappedByteBuffer,
			characterBuffer: MappedByteBuffer,
			wordTokenizer: FrozenSettingsTokenizer,
			characterTokenizer: FrozenCharacterSettingsTokenizer
		): SpecializedSettingsTfliteRuntime {
			var word: TfliteWordOperationHead? = null
			var character: TfliteCharacterSpeakerHead? = null
			try {
				word = TfliteWordOperationHead(createInterpreter(wordBuffer))
				character = TfliteCharacterSpeakerHead(createInterpreter(characterBuffer))
				return SpecializedSettingsTfliteRuntime(word, character, wordTokenizer, characterTokenizer)
			} catch (error: Throwable) {
				word?.close()
				character?.close()
				throw error
			}
		}

		private fun createInterpreter(buffer: MappedByteBuffer): Interpreter = Interpreter(
			buffer,
			Interpreter.Options().setNumThreads(2)
		)

		private fun mapAsset(context: Context, assetPath: String): MappedByteBuffer =
			context.assets.openFd(assetPath).use { descriptor ->
				FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
					channel.map(
						FileChannel.MapMode.READ_ONLY,
						descriptor.startOffset,
						descriptor.declaredLength
					)
				}
			}

		private fun mapFile(path: Path): MappedByteBuffer =
			FileChannel.open(path, StandardOpenOption.READ).use { channel ->
				channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
			}
	}
}

private abstract class TfliteSingleHead(
	private val interpreter: Interpreter,
	private val expectedLength: Int,
	private val outputName: String,
	private val outputSize: Int
) : AutoCloseable {
	init {
		interpreter.allocateTensors()
		require(SettingsTfliteContract.SIGNATURE_KEY in interpreter.signatureKeys) {
			"Frozen settings model is missing ${SettingsTfliteContract.SIGNATURE_KEY}"
		}
		require(SettingsTfliteContract.INPUT_NAME in interpreter.getSignatureInputs(SettingsTfliteContract.SIGNATURE_KEY))
		require(outputName in interpreter.getSignatureOutputs(SettingsTfliteContract.SIGNATURE_KEY)) {
			"Frozen settings model is missing signature output $outputName"
		}
		val input = interpreter.getInputTensorFromSignature(
			SettingsTfliteContract.INPUT_NAME,
			SettingsTfliteContract.SIGNATURE_KEY
		)
		val output = interpreter.getOutputTensorFromSignature(
			outputName,
			SettingsTfliteContract.SIGNATURE_KEY
		)
		require(input.shape().contentEquals(intArrayOf(1, expectedLength))) {
			"Unexpected settings input shape: ${input.shape().contentToString()}"
		}
		require(input.dataType() == DataType.INT32) { "Unexpected settings input dtype: ${input.dataType()}" }
		require(output.shape().contentEquals(intArrayOf(1, outputSize))) {
			"Unexpected $outputName output shape: ${output.shape().contentToString()}"
		}
		require(output.dataType() == DataType.FLOAT32) { "Unexpected $outputName output dtype: ${output.dataType()}" }
	}

	fun infer(tokenIds: IntArray): FloatArray {
		require(tokenIds.size == expectedLength) { "Unexpected token length ${tokenIds.size}" }
		val output = Array(1) { FloatArray(outputSize) }
		interpreter.runSignature(
			mapOf(SettingsTfliteContract.INPUT_NAME to arrayOf(tokenIds)),
			mutableMapOf<String, Any>(outputName to output),
			SettingsTfliteContract.SIGNATURE_KEY
		)
		return output[0].copyOf()
	}

	override fun close() = interpreter.close()
}

private class TfliteWordOperationHead(interpreter: Interpreter) :
	TfliteSingleHead(
		interpreter,
		SettingsTfliteContract.WORD_MAX_LEN,
		SettingsTfliteContract.OPERATION_OUTPUT_NAME,
		SettingOperation.entries.size
	), WordOperationHead {
	override fun inferOperation(tokenIds: IntArray): FloatArray = infer(tokenIds)
}

private class TfliteCharacterSpeakerHead(interpreter: Interpreter) :
	TfliteSingleHead(
		interpreter,
		SettingsTfliteContract.CHARACTER_MAX_LEN,
		SettingsTfliteContract.SPEAKER_OUTPUT_NAME,
		SpeakerChoice.entries.size
	), CharacterSpeakerHead {
	override fun inferSpeaker(tokenIds: IntArray): FloatArray = infer(tokenIds)
}
