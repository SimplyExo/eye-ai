package com.algorithmic_alliance.eyeaiapp.depth

import android.content.Context
import android.graphics.Bitmap
import android.util.Size
import java.io.File
import android.util.Log
import androidx.core.graphics.scale
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import uniffi.NativeLib.shutdownMetricDepthModel

/** All needed information to create and use a depth model */
class MetricDepthModelInfo(
	val name: String,
	val relativeDepthFileName: String
) {
	/** @return null if model type is not supported */
	fun createDepthModel(
		context: Context,
		skelDirectory: String,
		enableNpu: Boolean
	): MetricDepthModel {
		return MetricDepthModel(
			context,
			name,
			relativeDepthFileName,
			skelDirectory,
			enableNpu
		)
	}
}

class MetricDepthModel(
	context: Context,
	val name: String,
	relativeDepthFileName: String,
	skelDirectory: String,
	val enableNpu: Boolean
) : AutoCloseable {
	val inputDim: Size

	init {
		val relativeDepthModelData = context.assets.open(relativeDepthFileName).readBytes()

		uniffi.NativeLib.initMetricDepthModel(
			relativeDepthFileName, relativeDepthModelData, enableNpu, skelDirectory
		)

		val inputShape = uniffi.NativeLib.getMetricDepthModelInputShape()
		inputDim = if (inputShape.size == 4) {
			if (inputShape[0] != 1) {
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"first input shape of depth model should be 1, not ${inputShape[0]}"
				)
			}
			if (inputShape[3] != 3) {
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"depth model should take 3 channels for r,g,b, not ${inputShape[3]}"
				)
			}
			Size(inputShape[2], inputShape[1])
		} else {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"depth model input shape is not 4 dimensional, but ${inputShape.size} dimensional"
			)
			Size(256, 256)
		}

		val outputShape = uniffi.NativeLib.getMetricDepthModelOutputShape().toIntArray()
		val expectedOutputShape = intArrayOf(1, inputDim.height, inputDim.width, 1)
		if (!outputShape.contentEquals(expectedOutputShape)) {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"depth model has invalid output shape, expected [1, ${inputDim.height}, ${inputDim.width}, 1] but is [${outputShape}]"
			)
		}
	}

	override fun close() {
		/*NativeLib.*/shutdownMetricDepthModel()
	}

	/**
	 * @param input is not enforced to match [inputDim], but should be at least a bit larger
	 * @return relative depth for each pixel between 0.0f and 1.0f
	 */
	fun predictDepth(input: Bitmap): NativeLib.NativeFloatBuffer {
		val scaled = input.scale(inputDim.width, inputDim.height)
		val input = NativeLib.bitmapToRgbHwc255FloatArray(scaled)
		val output = NativeLib.NativeFloatBuffer(inputDim.width * inputDim.height)

		uniffi.NativeLib.runMetricDepthModelInference(
			input.asUniffiWrapper(),
			output.asUniffiWrapper(),
		)

		return output
	}
}