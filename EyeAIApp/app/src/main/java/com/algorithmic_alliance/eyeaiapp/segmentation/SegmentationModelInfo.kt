package com.algorithmic_alliance.eyeaiapp.segmentation

import android.content.Context

class SegmentationModelInfo(var tfliteFilename: String, var namesFilename: String, var size: Int) {
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
