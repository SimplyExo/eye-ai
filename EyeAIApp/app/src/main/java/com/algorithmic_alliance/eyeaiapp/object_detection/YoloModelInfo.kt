package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context

class YoloModelInfo(
	val name: String,
	val tfliteFilename: String,
	val namesFilename: String,
	val size: Int
) {
	fun getAsBytes(context: Context): ByteArray {
		context.assets.open(tfliteFilename).use { inputStream ->
			return inputStream.readBytes()
		}
	}

	fun readLinesFromAsset(context: Context): Array<String> {
		context.assets.open(namesFilename).bufferedReader().use { reader ->
			return reader.readLines().toTypedArray()
		}
	}
}
