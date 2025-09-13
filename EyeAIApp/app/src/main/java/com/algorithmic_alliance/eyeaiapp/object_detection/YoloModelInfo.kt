package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context

class YoloModelInfo(var filename: String, var size: Int) {
	fun getAsBytes(context: Context): ByteArray
	{
		context.assets.open(filename).use { inputStream ->
			return inputStream.readBytes()
		}
	}

	fun readLinesFromAsset(context: Context, filename: String): Array<String> {
		context.assets.open(filename).bufferedReader().use { reader ->
			return reader.readLines().toTypedArray()
		}
	}
}