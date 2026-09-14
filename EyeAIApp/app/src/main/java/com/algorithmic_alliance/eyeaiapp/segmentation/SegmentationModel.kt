package com.algorithmic_alliance.eyeaiapp.segmentation

import android.content.Context
import android.graphics.Bitmap
import androidx.core.graphics.scale
import com.algorithmic_alliance.eyeaiapp.NativeLib
import org.json.JSONObject

class SegmentationModel(var info: SegmentationModelInfo) {
	private lateinit var classes: List<String>
	lateinit var classColors: List<List<Int>>
	lateinit var classImportances: List<Float>

	var tensorWidth = 0
	var tensorHeight = 0

	private var initialized = false

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

		currentEnableNpu = enableNpu
		initialized = true
	}

	@Volatile
	private var currentEnableNpu: Boolean? = null

	@Synchronized
	fun runInference(frame: Bitmap): NativeLib.NativeIntBuffer? {
		if (!initialized) {			/*Log.e(
				"SEGMENTATION",
				"Tried to run Segmentation inference on uninitialized segmentation model, call create first!"
			)*/
			return null
		}

		val resizedBitmap = frame.scale(tensorWidth, tensorHeight, false)
		val input = NativeLib.bitmapToRgbHwc255FloatArray(resizedBitmap)
		val output = NativeLib.NativeIntBuffer(tensorWidth * tensorHeight)

		uniffi.NativeLib.runSegmentationOperation(input.asUniffiWrapper(), output.asUniffiWrapper())

		return output
	}
}
