package com.algorithmic_alliance.eyeaiapp.depth

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import androidx.core.graphics.scale
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib

class TfLiteDepthModel(
	context: Context,
	val fileName: String,
	val inputDim: Int,
	val normMean: FloatArray,
	val normStddev: FloatArray,
	enableProfiling: Boolean
) : DepthModel {
	private var formattedProfilerEntries: String? = null

	init {
		val modelData = context.assets.open(fileName).readBytes()

		val gpuDelegateCacheDirectory =
			createSerializedGpuDelegateCacheDirectory(context)
		val modelToken = getModelToken(context, fileName)

		// cleanup old cached gpu delegate files
		if (gpuDelegateCacheDirectory.exists()) {
			for (file in gpuDelegateCacheDirectory.listFiles()!!) {
				if (!file.name.contains(modelToken)) {
					try {
						Log.i(
							EyeAIApp.APP_LOG_TAG,
							"Deleting old gpu delegate cache file: ${file.name}"
						)
						file.delete()
					} catch (_: SecurityException) {
					}
				}
			}
		}

		NativeLib.initDepthTfLiteRuntime(
			modelData,
			gpuDelegateCacheDirectory.path,
			modelToken,
			enableProfiling
		)
	}

	override fun close() {
		NativeLib.shutdownDepthTfLiteRuntime()
	}

	override fun getName(): String = fileName

	override fun getInputSize(): Size = Size(inputDim, inputDim)

	override fun getFormattedProfilerEntries(): String? = formattedProfilerEntries

	override fun predictDepth(input: Bitmap): FloatArray {
		if (normMean.size != 3 || normStddev.size != 3) {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"normMean and normStddev should have exactly 3 elements for each rgb channel!"
			)
			return FloatArray(0)
		}

		val scaled = input.scale(inputDim, inputDim)
		val input = NativeLib.bitmapToRgbHwc255FloatArray(scaled)
		var output = FloatArray(inputDim * inputDim)

		val optionalFormattedProfilerEntries = NativeLib.runDepthTfLiteInference(
			input,
			output,
			normMean[0],
			normMean[1],
			normMean[2],
			normStddev[0],
			normStddev[1],
			normStddev[2]
		)
		if (optionalFormattedProfilerEntries != null)
			formattedProfilerEntries = optionalFormattedProfilerEntries

		return output
	}
}