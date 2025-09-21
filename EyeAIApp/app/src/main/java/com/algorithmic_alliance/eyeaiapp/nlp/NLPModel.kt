package com.algorithmic_alliance.eyeaiapp.nlp

import android.content.Context
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import org.tensorflow.lite.Interpreter

class NLPModel(var info: NLPModelInfo) {
	private var SEQUENCE_LENGTH = 250

	private var vocabFile = emptyArray<String>()

	enum class Classes {
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

	private var initialized = false

	private lateinit var interpreter: Interpreter


	fun create(context: Context) {
		// Creating NLP
		val modelBytes = info.loadModelFile(context)
		vocabFile = info.getVocab(context)

		interpreter = Interpreter(modelBytes)

		initialized = true
	}

	fun vectorizePrompt(prompt: String): FloatArray {
		val outputArray = FloatArray(SEQUENCE_LENGTH)

		prompt.split(" ").forEachIndexed { i, word ->
			if (i < outputArray.size) {
				var index = vocabFile.indexOf(word.lowercase())
				if (index == -1)
					index = 1
				outputArray[i] = index.toFloat()
			}
		}

		return outputArray
	}

	fun runInference(prompt: String): Classes {
		require(
			interpreter.getInputTensor(0).shape()[1] == SEQUENCE_LENGTH
		) {
			"input model: ${
				interpreter.getInputTensor(0).shape()[1]
			} input sequence: $SEQUENCE_LENGTH"
		}
		require(interpreter.getOutputTensor(0).shape()[1] == Classes.entries.size) {
			"output model: ${
				interpreter.getOutputTensor(0).shape()[1]
			} output classes: ${Classes.entries.size}"
		}

		val output = FloatArray(Classes.entries.size)
		interpreter.run(Array(1) { vectorizePrompt(prompt) }, Array(1) { output })

		var choice = 0
		var mostConfidence = 0.0f
		output.forEachIndexed { index, value ->
			if (value > mostConfidence) {
				mostConfidence = value
				choice = index
			}
		}

		return Classes.entries[choice]
	}
}
