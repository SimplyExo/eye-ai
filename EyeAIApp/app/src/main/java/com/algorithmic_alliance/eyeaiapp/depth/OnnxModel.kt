package com.algorithmic_alliance.eyeaiapp.depth

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import androidx.core.graphics.scale
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib

class OnnxModel(
	context: Context,
	val fileName: String,
	val inputDim: Int,
	val normMean: FloatArray,
	val normStddev: FloatArray
) : DepthModel {
	init {
		val modelData = context.assets.open(fileName).readBytes()

		NativeLib.initDepthOnnxRuntime(modelData)
	}

	override fun close() {
		NativeLib.shutdownDepthOnnxRuntime()
	}

	override fun getName(): String = fileName

	override fun getInputSize(): Size = Size(inputDim, inputDim)

	override fun predictDepth(input: Bitmap): FloatArray {
		if (normMean.size != 3 || normStddev.size != 3) {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"normMean and normStddev should have exactly 3 elements for each rgb channel!"
			)
			return FloatArray(0)
		}

		val scaled = input.scale(inputDim, inputDim)
		val input = NativeLib.bitmapToRgbChwFloatArray(scaled)
		var output = FloatArray(inputDim * inputDim)

		NativeLib.runDepthOnnxInference(
			input,
			output,
			normMean[0],
			normMean[1],
			normMean[2],
			normStddev[0],
			normStddev[1],
			normStddev[2]
		)

		return output
	}
}