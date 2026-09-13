package com.algorithmic_alliance.eyeaiapp.segmentation

import android.content.Context

class SegmentationModelInfo(var tfliteFilename: String, var namesJsonFilename: String, var size: Int) {
	fun getAsBytes(context: Context): ByteArray {
		context.assets.open(tfliteFilename).use { inputStream ->
			return inputStream.readBytes()
		}
	}

	fun readJsonFromAsset(context: Context): String {
		context.assets.open(namesJsonFilename).bufferedReader().use { reader ->
			return reader.readText()
		}
	}
}
