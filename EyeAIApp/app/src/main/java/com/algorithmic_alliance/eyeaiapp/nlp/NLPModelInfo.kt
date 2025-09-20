package com.algorithmic_alliance.eyeaiapp.nlp

import android.content.Context
import kotlin.collections.toTypedArray

class NLPModelInfo(var tfliteFilename: String) {
	fun getAsBytes(context: Context): ByteArray {
		context.assets.open(tfliteFilename).use { inputStream ->
			return inputStream.readBytes()
		}
	}

	fun getVocab(context: Context): Array<String> {
		context.assets.open("vocab.txt").bufferedReader().use { reader ->
			return reader.readLines().toTypedArray()
		}
	}
}
