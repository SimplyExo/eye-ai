package com.algorithmic_alliance.eyeaiapp.depth

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.util.Size
import java.io.File

/** All needed information to create and use a depth model */
class DepthModelInfo(
	val name: String,
	val fileName: String,
	val inputDim: Int,
	val normMean: FloatArray,
	val normStddev: FloatArray
) {
	/** @return null if model type is not supported */
	fun createDepthModel(context: Context): DepthModel? {
		if (normMean.size != 3 || normStddev.size != 3) return null

		if (fileName.endsWith(".tflite")) {
			return TfLiteDepthModel(
				context,
				fileName,
				inputDim,
				normMean,
				normStddev
			)
		} else if (fileName.endsWith(".onnx")) {
			return OnnxModel(context, fileName, inputDim, normMean, normStddev)
		}

		return null
	}
}

/** Base class that all depth estimation models implement */
interface DepthModel : AutoCloseable {
	fun getName(): String

	/**
	 * @param input is not enforced to match output of [getInputSize], but should be at least a bit
	 * larger
	 * @return relative depth for each pixel between 0.0f and 1.0f
	 */
	fun predictDepth(input: Bitmap): FloatArray

	/** @return preferred input image dimensions of the model */
	fun getInputSize(): Size
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
