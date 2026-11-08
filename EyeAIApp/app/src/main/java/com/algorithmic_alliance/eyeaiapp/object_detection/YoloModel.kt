package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import com.algorithmic_alliance.eyeaiapp.NativeLib
import uniffi.NativeLib.UniffiDetectedObject

class YoloModel(var info: YoloModelInfo) {
	private lateinit var labels: List<String>

	private var tensorWidth = 0
	private var tensorHeight = 0
	private var numChannel = 0
	private var numElements = 0

	private var initialized = false

	// Wiederverwendete Objekte: werden nur in create() neu angelegt, nicht pro Frame
	private var inputBuffer: NativeLib.NativeFloatBuffer? = null
	private var scaledBitmap: Bitmap? = null
	private var scaledCanvas: Canvas? = null
	private val srcRect = Rect()
	private val dstRect = Rect()

	@Volatile
	private var currentEnableNpu: Boolean? = null

	@Synchronized
	fun create(
		context: Context, skelDirectory: String, enableNpu: Boolean
	) {
		if (initialized && enableNpu == currentEnableNpu) {
			return
		}

		// Erstellen einer Yolo-Instanz
		val modelBytes = info.getAsBytes(context)
		labels = info.readLinesFromAsset(context).toList()

		val delegateCacheDirectory = NativeLib.createSerializedDelegateCacheDirectory(context)
		val modelToken = NativeLib.getModelToken(context, info.tfliteFilename)

		uniffi.NativeLib.initYoloRuntime(
			info.tfliteFilename,
			modelBytes,
			delegateCacheDirectory.absolutePath,
			modelToken,
			labels,
			enableNpu,
			skelDirectory
		)

		val inputShape = uniffi.NativeLib.getYoloInputShape()
		tensorWidth = inputShape[2]
		tensorHeight = inputShape[3]
		val outputShape = uniffi.NativeLib.getYoloOutputShape()
		numChannel = outputShape[1]
		numElements = outputShape[2]

		allocateBuffers()

		currentEnableNpu = enableNpu
		initialized = true
	}

	private fun allocateBuffers() {
		scaledBitmap?.recycle()
		val bitmap = Bitmap.createBitmap(tensorWidth, tensorHeight, Bitmap.Config.ARGB_8888)
		scaledBitmap = bitmap
		scaledCanvas = Canvas(bitmap)
		inputBuffer = NativeLib.NativeFloatBuffer(tensorWidth * tensorHeight * 3)
		dstRect.set(0, 0, tensorWidth, tensorHeight)
	}

	@Synchronized
	fun runInference(frame: Bitmap): Array<UniffiDetectedObject>? {
		if (!initialized) return null
		val canvas = scaledCanvas ?: return null
		val scaled = scaledBitmap ?: return null
		val input = inputBuffer ?: return null

		// Frame ohne neue Bitmap auf Tensorgröße skalieren (ohne Filter, wie bisher scale(..., false))
		srcRect.set(0, 0, frame.width, frame.height)
		canvas.drawBitmap(frame, srcRect, dstRect, null)

		// Direkt in den vorhandenen Buffer schreiben
		NativeLib.bitmapToRgbHwc255FloatArray(scaled, input.floatBuffer)

		return uniffi.NativeLib.runYoloOperation(input.asUniffiWrapper()).toTypedArray()
	}
}