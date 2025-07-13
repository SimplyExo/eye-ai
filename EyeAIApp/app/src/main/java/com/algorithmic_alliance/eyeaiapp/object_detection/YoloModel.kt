package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.compose.ui.geometry.Rect
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder

class YoloModel(var info: YoloModelInfo) {
	private var interpreter: Interpreter? = null
	private var labels = arrayOf(
		"person",
		"bicycle",
		"car",
		"motorcycle",
		"airplane",
		"bus",
		"train",
		"truck",
		"boat",
		"traffic light",
		"fire hydrant",
		"stop sign",
		"parking meter",
		"bench",
		"bird",
		"cat",
		"dog",
		"horse",
		"sheep",
		"cow",
		"elephant",
		"bear",
		"zebra",
		"giraffe",
		"backpack",
		"umbrella",
		"handbag",
		"tie",
		"suitcase",
		"frisbee",
		"skis",
		"snowboard",
		"sports ball",
		"kite",
		"baseball bat",
		"baseball glove",
		"skateboard",
		"surfboard",
		"tennis racket",
		"bottle",
		"wine glass",
		"cup",
		"fork",
		"knife",
		"spoon",
		"bowl",
		"banana",
		"apple",
		"sandwich",
		"orange",
		"broccoli",
		"carrot",
		"hot dog",
		"pizza",
		"donut",
		"cake",
		"chair",
		"couch",
		"potted plant",
		"bed",
		"dining table",
		"toilet",
		"tv",
		"laptop",
		"mouse",
		"remote",
		"keyboard",
		"cell phone",
		"microwave",
		"oven",
		"toaster",
		"sink",
		"refrigerator",
		"book",
		"clock",
		"vase",
		"scissors",
		"teddy bear",
		"hair drier",
		"toothbrush"
	)

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
		//NativeLib.initYoloRuntime(info.getAsBytes(context),
		//	createSerializedGpuDelegateCacheDirectory(context).path,
		//	getModelToken(context, info.filename))

		val modelBytes = info.getAsBytes(context)

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

	fun drawBoxesToBitmap(input: Bitmap, boxes: List<BoundingBox>): Bitmap
	{
		val mutableBitmap = input.copy(Bitmap.Config.ARGB_8888, true)
		val canvas = Canvas(mutableBitmap)

		// Boxen und labels zeichnen zeichnen
		for (box in boxes)
		{
			val left = box.x1 * input.width
			val top = box.y1 * input.height
			val right = box.x2 * input.width
			val bottom = box.y2 * input.height

			canvas.drawRect(left, top, right, bottom, paint_box);
			canvas.drawText(box.clsName, left, top, paint_text)
		}

		return mutableBitmap;
	}

	fun clear() {
		interpreter?.close()
		interpreter = null
	}

	fun runInference(frame: Bitmap): List<BoundingBox>? {
		interpreter ?: return null
		if (tensorWidth == 0) return null
		if (tensorHeight == 0) return null
		if (numChannel == 0) return null
		if (numElements == 0) return null

		var inferenceTime = SystemClock.uptimeMillis()

		val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

		val tensorImage = TensorImage(DataType.FLOAT32)
		tensorImage.load(resizedBitmap)
		val processedImage = imageProcessor.process(tensorImage)
		val imageBuffer = processedImage.buffer

		val output = TensorBuffer.createFixedSize(intArrayOf(1 , numChannel, numElements), OUTPUT_IMAGE_TYPE)
		interpreter?.run(imageBuffer, output.buffer)


		val bestBoxes = bestBox(output.floatArray)
		inferenceTime = SystemClock.uptimeMillis() - inferenceTime


		if (bestBoxes == null) {
			return bestBoxes;
		}

		return bestBoxes
	}

	private fun bestBox(array: FloatArray) : List<BoundingBox>? {

		val boundingBoxes = mutableListOf<BoundingBox>()

		for (c in 0 until numElements) {
			var maxConf = -1.0f
			var maxIdx = -1
			var j = 4
			var arrayIdx = c + numElements * j
			while (j < numChannel){
				if (array[arrayIdx] > maxConf) {
					maxConf = array[arrayIdx]
					maxIdx = j - 4
				}
				j++
				arrayIdx += numElements
			}

			if (maxConf > CONFIDENCE_THRESHOLD) {
				val clsName = labels[maxIdx]
				val cx = array[c] // 0
				val cy = array[c + numElements] // 1
				val w = array[c + numElements * 2]
				val h = array[c + numElements * 3]
				val x1 = cx - (w/2F)
				val y1 = cy - (h/2F)
				val x2 = cx + (w/2F)
				val y2 = cy + (h/2F)
				if (x1 < 0F || x1 > 1F) continue
				if (y1 < 0F || y1 > 1F) continue
				if (x2 < 0F || x2 > 1F) continue
				if (y2 < 0F || y2 > 1F) continue

				boundingBoxes.add(
					BoundingBox(
						x1 = x1, y1 = y1, x2 = x2, y2 = y2,
						cx = cx, cy = cy, w = w, h = h,
						cnf = maxConf, cls = maxIdx, clsName = clsName
					)
				)
			}
		}

		if (boundingBoxes.isEmpty()) return null

		return applyNMS(boundingBoxes)
	}

	private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
		val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
		val selectedBoxes = mutableListOf<BoundingBox>()

		while(sortedBoxes.isNotEmpty()) {
			val first = sortedBoxes.first()
			selectedBoxes.add(first)
			sortedBoxes.remove(first)

			val iterator = sortedBoxes.iterator()
			while (iterator.hasNext()) {
				val nextBox = iterator.next()
				val iou = calculateIoU(first, nextBox)
				if (iou >= IOU_THRESHOLD) {
					iterator.remove()
				}
			}
		}

		return selectedBoxes
	}

	private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
		val x1 = maxOf(box1.x1, box2.x1)
		val y1 = maxOf(box1.y1, box2.y1)
		val x2 = minOf(box1.x2, box2.x2)
		val y2 = minOf(box1.y2, box2.y2)
		val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
		val box1Area = box1.w * box1.h
		val box2Area = box2.w * box2.h
		return intersectionArea / (box1Area + box2Area - intersectionArea)
	}

	companion object {
		private const val INPUT_MEAN = 0f
		private const val INPUT_STANDARD_DEVIATION = 255f
		private val INPUT_IMAGE_TYPE = DataType.FLOAT32
		private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
		private const val CONFIDENCE_THRESHOLD = 0.3F
		private const val IOU_THRESHOLD = 0.5F
	}

	/*
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
	}*/
}