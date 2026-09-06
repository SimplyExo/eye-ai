package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.camera.TrackingEpoch
import java.io.File
import androidx.core.graphics.scale
import uniffi.NativeLib.UniffiDetectedObject

class YoloModel(var info: YoloModelInfo) {
	private val trackingSession = ObjectTrackingSession()
	private lateinit var labels: List<String>

	private var tensorWidth = 0
	private var tensorHeight = 0
	private var numChannel = 0
	private var numElements = 0

	@Volatile
	private var initialized = false

	val isReady: Boolean get() = initialized

	fun create(
		context: Context, skelDirectory: String,
		enableNpu: Boolean,
		onTrackerReplaced: () -> Unit,
	) = trackingSession.withModelLock {
		if (initialized && enableNpu == currentEnableNpu) {
			return@withModelLock
		}

		// Erstellen einer Yolo-Instanz
		initialized = false
		val modelBytes = info.getAsBytes(context)
		labels = info.readLinesFromAsset(context).toList()

		uniffi.NativeLib.initYoloRuntime(
			info.tfliteFilename, modelBytes, labels,
			enableNpu, skelDirectory
		)
		trackingSession.modelReplaced()
		// A real model replacement also replaces the native tracker/ID counter.
		// Publish its epoch boundary before another inference can take this lock.
		onTrackerReplaced()

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

	fun runInference(
		frame: Bitmap,
		trackingEpoch: TrackingEpoch,
		admit: () -> Boolean,
	): Array<UniffiDetectedObject>? = trackingSession.run(
		epoch = trackingEpoch,
		ready = { initialized },
		admit = admit,
		reset = { uniffi.NativeLib.resetObjectTracker() },
	) {
		uniffi.NativeLib.newObjectFrame()
		val resizedBitmap = frame.scale(tensorWidth, tensorHeight, false)
		val input = NativeLib.bitmapToRgbHwc255FloatArray(resizedBitmap)

		uniffi.NativeLib.runYoloOperation(input.asUniffiWrapper()).toTypedArray()
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
