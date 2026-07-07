package com.algorithmic_alliance.eyeaiapp.depth

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Size
import java.io.File
import android.util.Log
import androidx.core.graphics.scale
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.getLastAppUpdateTime
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

	private var reuseScaledBitmap: Bitmap? = null
	private var reuseInputBuffer: NativeLib.NativeFloatBuffer? = null
	private var reuseOutputBuffer: NativeLib.NativeFloatBuffer? = null
	private val scalePaint = Paint(Paint.FILTER_BITMAP_FLAG)

	init {
		val relativeDepthModelData = context.assets.open(relativeDepthFileName).readBytes()

		val gpuDelegateCacheDirectory =
			createSerializedGpuDelegateCacheDirectory(context)
		val relativeDepthModelToken = getModelToken(context, relativeDepthFileName)

		// cleanup old cached gpu delegate files
		if (gpuDelegateCacheDirectory.exists()) {
			for (file in gpuDelegateCacheDirectory.listFiles()!!) {
				if (file.name.contains(relativeDepthModelToken))
					continue

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
		val scaled = reuseScaledBitmap?.let {
			if (it.width == inputDim.width && it.height == inputDim.height) {
				Canvas(it).drawBitmap(input, null, Rect(0, 0, inputDim.width, inputDim.height), scalePaint)
				it
			} else null
		} ?: input.scale(inputDim.width, inputDim.height).also { reuseScaledBitmap = it }

		val inputBuf = NativeLib.bitmapToRgbHwc255FloatArray(scaled, reuseInputBuffer)
		reuseInputBuffer = inputBuf

		val output = reuseOutputBuffer?.takeIf { it.capacity >= inputDim.width * inputDim.height }
			?: NativeLib.NativeFloatBuffer(inputDim.width * inputDim.height).also { reuseOutputBuffer = it }

		uniffi.NativeLib.runMetricDepthModelInference(
			inputBuf.asUniffiWrapper(),
			output.asUniffiWrapper(),
		)

		return output
	}
}

fun createSerializedGpuDelegateCacheDirectory(context: Context): File {
	val gpuDelegateCacheDirectory = File(context.cacheDir, "gpu_delegate_cache")
	if (!gpuDelegateCacheDirectory.exists()) gpuDelegateCacheDirectory.mkdirs()
	return gpuDelegateCacheDirectory
}

/**
 * generates a unique token based on the model file name and last install/update time of this app
 */
fun getModelToken(context: Context, modelFilename: String): String {
	val lastUpdateTime = getLastAppUpdateTime(context)
	return "${modelFilename}_${lastUpdateTime}"
}
