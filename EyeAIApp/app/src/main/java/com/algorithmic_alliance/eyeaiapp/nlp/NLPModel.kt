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

	private var initialized = false

	fun create(context: Context) {
		// Erstellen einer Yolo-Instanz
		val modelBytes = info.getAsBytes(context)

		NativeLib.initNLPRuntime(
			modelBytes,
			createSerializedGpuDelegateCacheDirectory(context).path,
			getModelToken(context, info.tfliteFilename)
		)

		/*val inputShape = NativeLib.getNLPInputShape()
		tensorWidth = inputShape[1]
		tensorHeight = inputShape[2]
		val outputShape = NativeLib.getNLPOutputShape()
		numChannel = outputShape[1]
		numElements = outputShape[2]*/

		initialized = true
	}
}
