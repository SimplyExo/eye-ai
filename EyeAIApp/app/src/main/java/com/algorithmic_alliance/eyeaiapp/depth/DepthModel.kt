package com.algorithmic_alliance.eyeaiapp.depth

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.util.Size
import java.io.File
import android.util.Log
import androidx.core.graphics.scale
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib

/** All needed information to create and use a depth model */
class DepthModelInfo(
	val name: String,
	val fileName: String,
	val inputDim: Size
) {
	/** @return null if model type is not supported */
	fun createDepthModel(context: Context): DepthModel {
		return DepthModel(
			context,
			name,
			fileName,
			inputDim
		)
	}
}

class DepthModel(
	context: Context,
	val name: String,
	val fileName: String,
	val inputDim: Size
) : AutoCloseable {
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

		NativeLib.initDepthModel(
			modelData,
			gpuDelegateCacheDirectory.path,
			modelToken
		)
	}

	override fun close() {
		NativeLib.shutdownDepthModel()
	}

	/**
	 * @param input is not enforced to match [inputDim], but should be at least a bit larger
	 * @return relative depth for each pixel between 0.0f and 1.0f
	 */
	fun predictDepth(input: Bitmap): FloatArray {
		val scaled = input.scale(inputDim.width, inputDim.height)
		val input = NativeLib.bitmapToRgbHwc255FloatArray(scaled)
		var output = FloatArray(inputDim.width * inputDim.height)

		NativeLib.runDepthModelInference(
			input,
			output,
		)

		return output
	}
}

fun createSerializedGpuDelegateCacheDirectory(context: Context): File {
	val gpuDelegateCacheDirectory = File(context.cacheDir, "gpu_delegate_cache")
	if (!gpuDelegateCacheDirectory.exists()) gpuDelegateCacheDirectory.mkdirs()
	return gpuDelegateCacheDirectory
}

private fun getLastAppUpdateTime(context: Context): Long {
	try {
		val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
		return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
			packageInfo.lastUpdateTime
		} else {
			// Fallback
			File(context.packageCodePath).lastModified()
		}
	} catch (e: PackageManager.NameNotFoundException) {
		e.printStackTrace()
		return 0L
	}
}

/**
 * generates a unique token based on the model file name and last install/update time of this app
 */
fun getModelToken(context: Context, modelFilename: String): String {
	val lastUpdateTime = getLastAppUpdateTime(context)
	return "${modelFilename}_${lastUpdateTime}"
}
