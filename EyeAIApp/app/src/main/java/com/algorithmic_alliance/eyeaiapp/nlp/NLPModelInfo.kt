package com.algorithmic_alliance.eyeaiapp.nlp

import android.content.Context

class NLPModelInfo(var tfliteFilename: String) {
	fun getAsBytes(context: Context): ByteArray {
		context.assets.open(tfliteFilename).use { inputStream ->
			return inputStream.readBytes()
		}
	}
}