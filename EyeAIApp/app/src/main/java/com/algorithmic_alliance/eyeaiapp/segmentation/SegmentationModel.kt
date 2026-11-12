package com.algorithmic_alliance.eyeaiapp.segmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import com.algorithmic_alliance.eyeaiapp.NativeLib
import org.json.JSONObject

class SegmentationModel(var info: SegmentationModelInfo) {
	private lateinit var classes: List<String>
	lateinit var classColors: List<List<Int>>
	lateinit var classImportances: List<Float>

	var tensorWidth = 0
	var tensorHeight = 0

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

		val modelBytes = info.getAsBytes(context)
		val json = info.readJsonFromAsset(context)
		val jsonObject = JSONObject(json)

		val classes = mutableListOf<String>()
		val classColors = mutableListOf<List<Int>>()
		val classImportances = mutableListOf<Float>()

		val keys = jsonObject.keys()
		while (keys.hasNext()) {
			val className = keys.next()
			val classProperties = jsonObject.getJSONObject(className)
			val rgbArray = classProperties.getJSONArray("color")
			require(rgbArray.length() == 3)
			val importance = classProperties.getDouble("importance")
			classes.add(className)
			classColors.add(
				List(rgbArray.length()) { index ->
					rgbArray.getInt(index)
				}
			)
			classImportances.add(importance.toFloat())
		}
		this.classes = classes
		this.classColors = classColors
		this.classImportances = classImportances

		val delegateCacheDirectory = NativeLib.createSerializedDelegateCacheDirectory(context)
		val modelToken = NativeLib.getModelToken(context, info.tfliteFilename)

		uniffi.NativeLib.initSegmentationRuntime(
			info.tfliteFilename,
			modelBytes,
			delegateCacheDirectory.absolutePath,
			modelToken,
			classes,
			enableNpu,
			skelDirectory
		)

		val inputShape = uniffi.NativeLib.getSegmentationInputShape()
		tensorWidth = inputShape[2]
		tensorHeight = inputShape[3]

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

	/** Published masks retain their own storage for asynchronous consumers. */
	@Synchronized
	fun runInference(frame: Bitmap): NativeLib.NativeIntBuffer? {
		if (!initialized) return null
		val canvas = scaledCanvas ?: return null
		val scaled = scaledBitmap ?: return null
		val input = inputBuffer ?: return null
		val output = NativeLib.NativeIntBuffer(tensorWidth * tensorHeight)

		// Frame ohne neue Bitmap auf Tensorgröße skalieren
		srcRect.set(0, 0, frame.width, frame.height)
		canvas.drawBitmap(frame, srcRect, dstRect, null)

		// In den vorhandenen Buffer schreiben (Position zurücksetzen!)
		input.floatBuffer.clear()
		NativeLib.bitmapToRgbHwc255FloatArray(scaled, input.floatBuffer)

		uniffi.NativeLib.runSegmentationOperation(input.asUniffiWrapper(), output.asUniffiWrapper())
		return output
	}
}