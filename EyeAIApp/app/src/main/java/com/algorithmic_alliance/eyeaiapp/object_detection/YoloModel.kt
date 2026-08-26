package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context
import android.graphics.Bitmap
import com.algorithmic_alliance.eyeaiapp.NativeLib
import androidx.core.graphics.scale
import uniffi.NativeLib.UniffiDetectedObject

class YoloModel(var info: YoloModelInfo) {
	private lateinit var labels: List<String>

	private var tensorWidth = 0
	private var tensorHeight = 0
	private var numChannel = 0
	private var numElements = 0

	private var initialized = false

	fun create(
		context: Context, skelDirectory: String,
		enableNpu: Boolean
	) {
		// Erstellen einer Yolo-Instanz
		val modelBytes = info.getAsBytes(context)
		labels = info.readLinesFromAsset(context).toList()

		val delegateCacheDirectory =
			NativeLib.createSerializedDelegateCacheDirectory(context)
		val modelToken = NativeLib.getModelToken(context, info.tfliteFilename)

		uniffi.NativeLib.initYoloRuntime(
			info.tfliteFilename, modelBytes,delegateCacheDirectory.absolutePath, modelToken, labels,
			 enableNpu, skelDirectory
		)

		val inputShape = uniffi.NativeLib.getYoloInputShape()
		tensorWidth = inputShape[1]
		tensorHeight = inputShape[2]
		val outputShape = uniffi.NativeLib.getYoloOutputShape()
		numChannel = outputShape[1]
		numElements = outputShape[2]

		initialized = true
	}

	fun runInference(frame: Bitmap): Array<UniffiDetectedObject>? {
		if (!initialized) {
			/*Log.e(
				"YOLO",
				"Tried to run YOLO inference on uninitialized yolo model, call create first!"
			)*/
			return null
		}

		val resizedBitmap = frame.scale(tensorWidth, tensorHeight, false)
		val input = NativeLib.bitmapToRgbHwc255FloatArray(resizedBitmap)

		return uniffi.NativeLib.runYoloOperation(input.asUniffiWrapper()).toTypedArray()
	}
}