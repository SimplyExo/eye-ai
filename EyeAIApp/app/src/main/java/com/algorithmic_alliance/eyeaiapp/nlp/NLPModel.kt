package com.algorithmic_alliance.eyeaiapp.nlp

import android.content.Context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

class NLPModel(initialInfo: NLPModelInfo) : AutoCloseable {
	var info: NLPModelInfo = initialInfo
		private set

	enum class NLPClasses {
		TEXT_RECOGNITION,
		OBJECT_DETECTION,
		CHANGE_SPEECH_SPEED,
		CHANGE_SPEAKER,
		REDIRECT_TO_LLM,
		OPEN_SETTINGS,
		SET_FREQUENCY,
		SET_BPS,
		MEASURE_DISTANCE,
		ABORT
	}

	data class NLPResults(
		val TEXT_RECOGNITION: Float,
		val OBJECT_DETECTION: Float,
		val CHANGE_SPEECH_SPEED: Float,
		val CHANGE_SPEAKER: Float,
		val REDIRECT_TO_LLM: Float,
		val OPEN_SETTINGS: Float,
		val SET_FREQUENCY: Float,
		val SET_BPS: Float,
		val MEASURE_DISTANCE: Float,
		val ABORT: Float
	)

	private var interpreter: Interpreter? = null
	private var tokenizer: IntentTokenizer? = null

	val isInitialized: Boolean
		@Synchronized get() = interpreter != null && tokenizer != null

	@Synchronized
	fun create(context: Context, modelInfo: NLPModelInfo = info) {
		val loadedTokenizer = modelInfo.loadTokenizer(context)
		val expectedLabels = NLPClasses.entries.map { it.name }
		require(loadedTokenizer.labels == expectedLabels) {
			"NLP labels do not match the app's intent order"
		}

		val newInterpreter = Interpreter(
			modelInfo.loadModelFile(context),
			Interpreter.Options().setNumThreads(2)
		)
		try {
			validateModel(newInterpreter, loadedTokenizer.tokenizer)
		} catch (error: Throwable) {
			newInterpreter.close()
			throw error
		}

		interpreter?.close()
		interpreter = newInterpreter
		tokenizer = loadedTokenizer.tokenizer
		info = modelInfo
	}

	@Synchronized
	fun vectorizePrompt(prompt: String): IntArray =
		requireNotNull(tokenizer) { "NLP model is not initialized" }.encode(prompt)

	@Synchronized
	fun runInferenceWithAllResults(prompt: String): NLPResults {
		val activeInterpreter = requireNotNull(interpreter) { "NLP model is not initialized" }
		val input = vectorizePrompt(prompt)
		val output = FloatArray(NLPClasses.entries.size)
		activeInterpreter.run(arrayOf(input), arrayOf(output))

		return NLPResults(
			TEXT_RECOGNITION = output[0],
			OBJECT_DETECTION = output[1],
			CHANGE_SPEECH_SPEED = output[2],
			CHANGE_SPEAKER = output[3],
			REDIRECT_TO_LLM = output[4],
			OPEN_SETTINGS = output[5],
			SET_FREQUENCY = output[6],
			SET_BPS = output[7],
			MEASURE_DISTANCE = output[8],
			ABORT = output[9]
		)
	}

	@Synchronized
	fun runInference(prompt: String): NLPClasses {
		val results = runInferenceWithAllResults(prompt)
		val probabilities = floatArrayOf(
			results.TEXT_RECOGNITION,
			results.OBJECT_DETECTION,
			results.CHANGE_SPEECH_SPEED,
			results.CHANGE_SPEAKER,
			results.REDIRECT_TO_LLM,
			results.OPEN_SETTINGS,
			results.SET_FREQUENCY,
			results.SET_BPS,
			results.MEASURE_DISTANCE,
			results.ABORT
		)
		val bestIndex = probabilities.indices.maxBy { probabilities[it] }
		return NLPClasses.entries[bestIndex]
	}

	@Synchronized
	override fun close() {
		interpreter?.close()
		interpreter = null
		tokenizer = null
	}

	private fun validateModel(interpreter: Interpreter, tokenizer: IntentTokenizer) {
		val inputTensor = interpreter.getInputTensor(0)
		val outputTensor = interpreter.getOutputTensor(0)
		require(inputTensor.shape().contentEquals(intArrayOf(1, tokenizer.maxLength))) {
			"Unexpected NLP input shape: ${inputTensor.shape().contentToString()}"
		}
		require(inputTensor.dataType() == DataType.INT32) {
			"Unexpected NLP input type: ${inputTensor.dataType()}"
		}
		require(outputTensor.shape().contentEquals(intArrayOf(1, NLPClasses.entries.size))) {
			"Unexpected NLP output shape: ${outputTensor.shape().contentToString()}"
		}
		require(outputTensor.dataType() == DataType.FLOAT32) {
			"Unexpected NLP output type: ${outputTensor.dataType()}"
		}
	}
}
