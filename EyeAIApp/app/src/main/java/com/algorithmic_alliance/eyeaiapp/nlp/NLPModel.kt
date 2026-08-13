package com.algorithmic_alliance.eyeaiapp.nlp

import android.content.Context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

class NLPModel(initialInfo: NLPModelInfo) : AutoCloseable {
	var info: NLPModelInfo = initialInfo
		private set

	private var interpreter: Interpreter? = null
	private var tokenizer: IntentTokenizer? = null

	val isInitialized: Boolean
		@Synchronized get() = interpreter != null && tokenizer != null

	@Synchronized
	fun create(context: Context, modelInfo: NLPModelInfo = info) {
		val loadedTokenizer = modelInfo.loadTokenizer(context)
		val expectedLabels = Intent.CLASS_ORDER.map { it.name }
		require(loadedTokenizer.labels == expectedLabels) {
			"NLP V2 labels do not match Intent.CLASS_ORDER"
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

	/** Runs exactly one NLP V2 inference and preserves the unchanged input text. */
	@Synchronized
	fun classify(originalText: String): IntentResult {
		val activeInterpreter = requireNotNull(interpreter) { "NLP model is not initialized" }
		val activeTokenizer = requireNotNull(tokenizer) { "NLP tokenizer is not initialized" }
		val input = activeTokenizer.encode(originalText)
		val probabilities = FloatArray(Intent.CLASS_ORDER.size)
		activeInterpreter.run(arrayOf(input), arrayOf(probabilities))

		return IntentResult.fromProbabilities(originalText, probabilities)
	}

	@Synchronized
	override fun close() {
		interpreter?.close()
		interpreter = null
		tokenizer = null
	}

	private fun validateModel(interpreter: Interpreter, tokenizer: IntentTokenizer) {
		require(interpreter.inputTensorCount == 1) {
			"Expected exactly one NLP V2 input tensor, got ${interpreter.inputTensorCount}"
		}
		require(interpreter.outputTensorCount == 1) {
			"Expected exactly one NLP V2 output tensor, got ${interpreter.outputTensorCount}"
		}
		require(tokenizer.maxLength == INPUT_LENGTH) {
			"Unexpected NLP V2 tokenizer length: ${tokenizer.maxLength}"
		}
		val inputTensor = interpreter.getInputTensor(0)
		val outputTensor = interpreter.getOutputTensor(0)
		require(inputTensor.shape().contentEquals(intArrayOf(1, INPUT_LENGTH))) {
			"Unexpected NLP input shape: ${inputTensor.shape().contentToString()}"
		}
		require(inputTensor.dataType() == DataType.INT32) {
			"Unexpected NLP input type: ${inputTensor.dataType()}"
		}
		require(outputTensor.shape().contentEquals(intArrayOf(1, Intent.CLASS_ORDER.size))) {
			"Unexpected NLP output shape: ${outputTensor.shape().contentToString()}"
		}
		require(outputTensor.dataType() == DataType.FLOAT32) {
			"Unexpected NLP output type: ${outputTensor.dataType()}"
		}
	}

	companion object {
		const val INPUT_LENGTH = 24
	}
}
