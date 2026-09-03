package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import com.algorithmic_alliance.eyeaiapp.NativeLib
import java.io.File
import androidx.core.graphics.scale
import uniffi.NativeLib.UniffiDetectedObject

class YoloModel(var info: YoloModelInfo) {
	private lateinit var labels: List<String>

	private var tensorWidth = 0
	private var tensorHeight = 0
	private var numChannel = 0
	private var numElements = 0

	private var initialized = false

	@Synchronized
	fun create(
		context: Context, skelDirectory: String,
		enableNpu: Boolean
	) {
		if (initialized && enableNpu == currentEnableNpu) {
			return
		}

		// Erstellen einer Yolo-Instanz
		val modelBytes = info.getAsBytes(context)
		labels = info.readLinesFromAsset(context).toList()

		uniffi.NativeLib.initYoloRuntime(
			info.tfliteFilename, modelBytes, labels,
			enableNpu, skelDirectory
		)

		val inputShape = uniffi.NativeLib.getYoloInputShape()
		tensorWidth = inputShape[1]
		tensorHeight = inputShape[2]
		val outputShape = uniffi.NativeLib.getYoloOutputShape()
		numChannel = outputShape[1]
		numElements = outputShape[2]

		currentEnableNpu = enableNpu
		initialized = true
	}

	@Volatile
	private var currentEnableNpu: Boolean? = null

	@Synchronized
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
