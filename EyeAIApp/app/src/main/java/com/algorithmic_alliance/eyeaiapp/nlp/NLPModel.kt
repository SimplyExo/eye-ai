package com.algorithmic_alliance.eyeaiapp.nlp

import android.content.Context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

class NLPModel(initialInfo: NLPModelInfo) : AutoCloseable {
	var info: NLPModelInfo = initialInfo
		private set

	private var interpreter: Interpreter? = null
	private var tokenizer: IntentTokenizer? = null
    private var modeInterpreter: Interpreter? = null
    private var modeThreshold = ModeRouterDecision.THRESHOLD

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
			modelInfo.loadModelFile(context), Interpreter.Options().setNumThreads(2)
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
        modeInterpreter?.close()
        modeInterpreter = null
        try {
            val config = org.json.JSONObject(context.assets.open("nlp-v2/mode-router/config.json").bufferedReader().use { it.readText() })
            val labels = config.getJSONArray("labels")
            require((0 until labels.length()).map { labels.getString(it) } == ModeRouterDecision.LABELS)
            require(modelInfo.tokenizerAssetDirectory == "nlp-v2/tokenizers/T2")
            modeThreshold = config.getDouble("threshold").toFloat()
            require(modeThreshold == ModeRouterDecision.THRESHOLD)
            val buffer = modelInfo.copy(tfliteAssetPath = "nlp-v2/mode-router/model.tflite").loadModelFile(context)
            val digest = java.security.MessageDigest.getInstance("SHA-256")
            digest.update(buffer.asReadOnlyBuffer())
            require(digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) } == config.getString("model_sha256"))
            val router = Interpreter(buffer, Interpreter.Options().setNumThreads(2))
            try {
                require(router.inputTensorCount == 1 && router.outputTensorCount == 1)
                require(router.getInputTensor(0).shape().contentEquals(intArrayOf(1, INPUT_LENGTH)))
                require(router.getInputTensor(0).dataType() == DataType.INT32)
                require(router.getOutputTensor(0).shape().contentEquals(intArrayOf(1, 4)))
                require(router.getOutputTensor(0).dataType() == DataType.FLOAT32)
                modeInterpreter = router
            } catch (error: Throwable) { router.close(); throw error }
        } catch (error: Exception) {
            android.util.Log.e("EyeAI", "Mode router unavailable; using legacy intent classifier", error)
        }
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

    /** Returns null for NO_MODE, low confidence or an unavailable router. */
    @Synchronized
    fun classifyMode(originalText: String): com.algorithmic_alliance.eyeaiapp.audio.AudioOutputMode? {
        val router = modeInterpreter ?: return null
        val encoder = tokenizer ?: return null
        val probabilities = FloatArray(4)
        router.run(arrayOf(encoder.encode(originalText)), arrayOf(probabilities))
        return ModeRouterDecision.select(probabilities, modeThreshold)
    }

	@Synchronized
	override fun close() {
		interpreter?.close()
		interpreter = null
        modeInterpreter?.close()
        modeInterpreter = null
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
