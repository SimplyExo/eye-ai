package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Build
import android.os.SystemClock
import com.algorithmic_alliance.eyeaiapp.NativeLib
import java.io.File
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.core.graphics.scale

class YoloModel(var info: YoloModelInfo) {
	private var interpreter: Interpreter? = null
	private lateinit var labels: Array<String>

	private var tensorWidth = 0
	private var tensorHeight = 0
	private var numChannel = 0
	private var numElements = 0

	private val imageProcessor = ImageProcessor.Builder()
		.add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
		.add(CastOp(INPUT_IMAGE_TYPE))
		.build()

	private val paint_box = Paint().apply {
		color = Color.RED
		style = Paint.Style.STROKE
		strokeWidth = 2f
	}

	val paint_text = Paint().apply {
		color = Color.RED
		strokeWidth = 1f
	}

	fun create(context: Context)
	{
		// Erstellen einer Yolo-Instanz
		val modelBytes = info.getAsBytes(context)
		labels = info.readLinesFromAsset(context, "coco.names")

		NativeLib.initYoloRuntime(modelBytes, labels,
			createSerializedGpuDelegateCacheDirectory(context).path,
			getModelToken(context, info.filename))

		val model = ByteBuffer.allocateDirect(modelBytes.size)
			.order(ByteOrder.nativeOrder())
		model.put(modelBytes)
		model.rewind()

		val options = Interpreter.Options()
		options.numThreads = 4
		interpreter = Interpreter(model, options)

		val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
		val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return

		tensorWidth = inputShape[1]
		tensorHeight = inputShape[2]
		numChannel = outputShape[1]
		numElements = outputShape[2]
	}

	fun clear() {
		interpreter?.close()
		interpreter = null
	}

	fun runInference(frame: Bitmap): Array<BoundingBox>? {
		interpreter ?: return null
		if (tensorWidth == 0) return null
		if (tensorHeight == 0) return null
		if (numChannel == 0) return null
		if (numElements == 0) return null

		var inferenceTime = SystemClock.uptimeMillis()

		val resizedBitmap = frame.scale(tensorWidth, tensorHeight, false)

		val tensorImage = TensorImage(DataType.FLOAT32)
		tensorImage.load(resizedBitmap)

        val buffer = tensorImage.buffer
        buffer.rewind()

        val floatArray = FloatArray(buffer.remaining() / 4)
        buffer.asFloatBuffer().get(floatArray)

		val output = NativeLib.runYoloOperation(floatArray, numElements, numChannel);

		val bestBoxes = output
		inferenceTime = SystemClock.uptimeMillis() - inferenceTime

		return bestBoxes
	}

	companion object {
		private const val INPUT_MEAN = 0f
		private const val INPUT_STANDARD_DEVIATION = 255f
		private val INPUT_IMAGE_TYPE = DataType.FLOAT32
		private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
		private const val CONFIDENCE_THRESHOLD = 0.5F
		private const val IOU_THRESHOLD = 0.5F
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