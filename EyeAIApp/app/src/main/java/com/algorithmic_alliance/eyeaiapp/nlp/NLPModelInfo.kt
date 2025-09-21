package com.algorithmic_alliance.eyeaiapp.nlp

import android.content.Context
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.collections.toTypedArray

class NLPModelInfo(var tfliteFilename: String) {
	fun loadModelFile(context: Context): MappedByteBuffer {
		val fileDescriptor = context.assets.openFd(tfliteFilename)
		FileInputStream(fileDescriptor.fileDescriptor).channel.use { channel ->
			return channel.map(
				FileChannel.MapMode.READ_ONLY,
				fileDescriptor.startOffset,
				fileDescriptor.declaredLength
			)
		}
	}

	fun getVocab(context: Context): Array<String> {
		context.assets.open("vocab.txt").bufferedReader().use { reader ->
			return reader.readLines().toTypedArray()
		}
	}
}
