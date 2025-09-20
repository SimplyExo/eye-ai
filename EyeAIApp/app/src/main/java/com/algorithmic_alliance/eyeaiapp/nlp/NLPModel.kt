package com.algorithmic_alliance.eyeaiapp.nlp

import android.content.Context
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.depth.createSerializedGpuDelegateCacheDirectory
import com.algorithmic_alliance.eyeaiapp.depth.getModelToken

class NLPModel(var info: NLPModelInfo) {
	private var tensorWidth = 0
	private var tensorHeight = 0
	private var numChannel = 0
	private var numElements = 0
	private var SEQUENCE_LENGTH = 250

	private var vocabFile = emptyArray<String>();

	private var initialized = false


	fun create(context: Context, skelDirectory: String) {
		// Erstellen einer NLP-Instanz
		val modelBytes = info.getAsBytes(context)
		vocabFile = info.getVocab(context)

		NativeLib.initNLPRuntime(
			modelBytes,
			createSerializedGpuDelegateCacheDirectory(context).path,
			getModelToken(context, info.tfliteFilename), skelDirectory
		)

		initialized = true
	}

	fun vectorizePrompt(prompt: String): FloatArray {
		val output_array = FloatArray(SEQUENCE_LENGTH)

		vocabFile.forEachIndexed { index, word ->
			var res = vocabFile.indexOf(word)

			if (res == -1)
				res = 1

			output_array[index] = res.toFloat()
		}

		return output_array
	}

	fun runInference(prompt: String) {
		NativeLib.runNLPOperation(vectorizePrompt(prompt))
	}

	fun inputShape(): IntArray {
		return NativeLib.getNLPInputShape()
	}

	fun outputShape(): IntArray {
		return NativeLib.getNLPOutputShape()
	}
}
