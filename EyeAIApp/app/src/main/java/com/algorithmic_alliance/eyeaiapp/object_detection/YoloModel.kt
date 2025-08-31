package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.NativeLib
import java.io.File
import androidx.core.graphics.scale
import com.algorithmic_alliance.eyeaiapp.ProfilingFrameType
import org.json.*

class YoloModel(var info: YoloModelInfo) {
	private lateinit var labels: Array<String>

	private var tensorWidth = 0
	private var tensorHeight = 0
	private var numChannel = 0
	private var numElements = 0

	private var initialized = false

	fun create(context: Context) {
		// Erstellen einer Yolo-Instanz
		val modelBytes = info.getAsBytes(context)
		labels = info.readLinesFromAsset(context, "coco.names")

		NativeLib.initYoloRuntime(
			modelBytes, labels,
			createSerializedGpuDelegateCacheDirectory(context).path,
			getModelToken(context, info.filename)
		)

		val inputShape = NativeLib.getYoloInputShape()
		tensorWidth = inputShape[1]
		tensorHeight = inputShape[2]
		val outputShape = NativeLib.getYoloOutputShape()
		numChannel = outputShape[1]
		numElements = outputShape[2]

		initialized = true
	}

	fun runInference(frame: Bitmap): Array<BoundingBox>? {
		if (!initialized) {
			Log.e(
				"YOLO",
				"Tried to run YOLO inference on uninitialized yolo model, call create first!"
			)
			return null
		}

		val resizedBitmap = frame.scale(tensorWidth, tensorHeight, false)
		val input = NativeLib.bitmapToRgbHwc255FloatArray(resizedBitmap, ProfilingFrameType.Object)

		val json_string = NativeLib.runYoloOperation(input)

		// Wenn string leer ist --> Keine Objekte erkannt!
		if (json_string == "null")
			return emptyArray()

		val json_object = JSONObject(json_string)
		val boxes = json_object.getJSONArray("bounding_boxes")

		val bestBoxes = ArrayList<BoundingBox>()
		for (i in 0 until boxes.length()) {
			val b = boxes.getJSONObject(i)

			val boundingBox = BoundingBox(
				b.getDouble("x1").toFloat(), b.getDouble("y1").toFloat(),
				b.getDouble("x2").toFloat(), b.getDouble("y2").toFloat(),
				b.getDouble("cx").toFloat(), b.getDouble("cy").toFloat(),
				b.getDouble("w").toFloat(), b.getDouble("h").toFloat(),
				b.getDouble("cnf").toFloat(), b.getInt("cls"),
				b.getString("clsName")
			)

			bestBoxes.add(boundingBox)
		}

		return bestBoxes.toTypedArray()
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