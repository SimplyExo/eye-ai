package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context

class YoloModelInfo(var filename: String, var size: Int) {
	fun getAsBytes(context: Context): ByteArray
	{
		context.assets.open(filename).use { inputStream ->
			return inputStream.readBytes()
		}
	}

	fun readLinesFromAsset(context: Context): Array<String> {
		val splitUp = filename.split(".")
		val annotations = splitUp.take(splitUp.size-1).joinToString(".") + "names"
		context.assets.open(annotations).bufferedReader().use { reader ->
			return reader.readLines().toTypedArray()
		}
	}
}