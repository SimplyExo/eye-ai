package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.SystemClock
import com.algorithmic_alliance.eyeaiapp.NativeLib
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.core.graphics.scale
import java.lang.annotation.Native

class YoloModel(var info: YoloModelInfo) {
	private lateinit var labels: Array<String>

	private var tensorWidth = 0
	private var tensorHeight = 0
	private var numChannel = 0
	private var numElements = 0

	fun create(context: Context) {
		// Erstellen einer Yolo-Instanz
		val modelBytes = info.getAsBytes(context)
		labels = info.readLinesFromAsset(context, "coco.names")

		NativeLib.initYoloRuntime(
			modelBytes, labels,
			createSerializedGpuDelegateCacheDirectory(context).path,
			getModelToken(context, info.filename)
		)

		tensorWidth = NativeLib.getInputShape()[1]
		tensorHeight = NativeLib.getInputShape()[2]
		numChannel = NativeLib.getOutputShape()[1]
		numElements = NativeLib.getOutputShape()[2]
	}

	fun clear() {
		// Not implemented yet!
	}

	fun runInference(frame: Bitmap): Array<BoundingBox>? {
		val resizedBitmap = frame.scale(tensorWidth, tensorHeight, false)
		val input = NativeLib.bitmapToRgbHwc255FloatArray(resizedBitmap)

		val output = NativeLib.runYoloOperation(input);

		val bestBoxes = output

		return bestBoxes
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

	private fun getModelToken(context: Context, modelFilename: String): String {
		val lastUpdateTime = getLastAppUpdateTime(context)
		return "${modelFilename}_${lastUpdateTime}"
	}
}