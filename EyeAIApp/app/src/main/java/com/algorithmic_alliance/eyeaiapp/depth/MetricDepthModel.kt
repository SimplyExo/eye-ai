package com.algorithmic_alliance.eyeaiapp.depth

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib

/** All needed information to create and use a depth model */
class MetricDepthModelInfo(
	val name: String, val relativeDepthFileName: String
) {
	/** @return null if model type is not supported */
	fun createDepthModel(
		context: Context, skelDirectory: String, enableNpu: Boolean
	): MetricDepthModel {
		return MetricDepthModel(
			context, name, relativeDepthFileName, skelDirectory, enableNpu
		)
	}
}

/**
 * The legacy metric-depth output and the unmodified MiDaS relative-depth output from one
 * native inference pass.  Keeping both is essential: the legacy output remains the current
 * app behavior, while REL2ABS consumes the raw MiDaS representation it was trained on.
 */
data class DepthModelOutputs(
	val legacyMetricDepth: NativeLib.NativeFloatBuffer,
	val rawRelativeDepth: NativeLib.NativeFloatBuffer,
)

class MetricDepthModel(
	context: Context,
	val name: String,
	relativeDepthFileName: String,
	skelDirectory: String,
	val enableNpu: Boolean
) {
	val inputDim: Size

	// Wiederverwendete Objekte: werden nur einmal im init angelegt, nicht pro Frame.
	// Wichtig: Diese Felder müssen VOR dem init-Block stehen.
	private val inputBuffer: NativeLib.NativeFloatBuffer
	private val scaledBitmap: Bitmap
	private val scaledCanvas: Canvas
	private val outputRing: Array<DepthModelOutputs>
	private var outputIndex = 0
	private val srcRect = Rect()
	private val dstRect = Rect()

	// Entspricht dem bisherigen Bitmap.scale(w, h) (filter = true)
	private val scalePaint = Paint(Paint.FILTER_BITMAP_FLAG)

	init {
		val relativeDepthModelData = context.assets.open(relativeDepthFileName).readBytes()

		val delegateCacheDirectory = NativeLib.createSerializedDelegateCacheDirectory(context)
		val relativeDepthModelToken = NativeLib.getModelToken(context, relativeDepthFileName)

		uniffi.NativeLib.initMetricDepthModel(
			relativeDepthFileName,
			relativeDepthModelData,
			delegateCacheDirectory.absolutePath,
			relativeDepthModelToken,
			enableNpu,
			skelDirectory
		)

		val inputShape = uniffi.NativeLib.getMetricDepthModelInputShape()
		inputDim = if (inputShape.size == 4) {
			if (inputShape[0] != 1) {
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"first input shape of depth model should be 1, not ${inputShape[0]}"
				)
			}
			if (inputShape[3] != 3) {
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"depth model should take 3 channels for r,g,b, not ${inputShape[3]}"
				)
			}
			Size(inputShape[2], inputShape[1])
		} else {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"depth model input shape is not 4 dimensional, but ${inputShape.size} dimensional"
			)
			Size(256, 256)
		}

		val outputShape = uniffi.NativeLib.getMetricDepthModelOutputShape().toIntArray()
		val expectedOutputShape = intArrayOf(1, inputDim.height, inputDim.width, 1)
		if (!outputShape.contentEquals(expectedOutputShape)) {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"depth model has invalid output shape, expected [1, ${inputDim.height}, ${inputDim.width}, 1] but is [${outputShape.contentToString()}]"
			)
		}

		val pixelCount = inputDim.width * inputDim.height
		inputBuffer = NativeLib.NativeFloatBuffer(pixelCount * 3)
		scaledBitmap = Bitmap.createBitmap(inputDim.width, inputDim.height, Bitmap.Config.ARGB_8888)
		scaledCanvas = Canvas(scaledBitmap)
		dstRect.set(0, 0, inputDim.width, inputDim.height)

		// Die Outputs gehen an andere Threads (UI, Audio, Rel2Abs) und dürfen nicht sofort
		// überschrieben werden. Deshalb ein Ring aus mehreren Paaren statt nur einem.
		outputRing = Array(OUTPUT_RING_SIZE) {
			DepthModelOutputs(
				legacyMetricDepth = NativeLib.NativeFloatBuffer(pixelCount),
				rawRelativeDepth = NativeLib.NativeFloatBuffer(pixelCount),
			)
		}
	}

	/**
	 * @param input is not enforced to match [inputDim], but should be at least a bit larger
	 * @return legacy metric depth plus raw MiDaS relative depth from the same backbone pass.
	 * Die zurückgegebenen Buffer werden nach [OUTPUT_RING_SIZE] weiteren Aufrufen überschrieben.
	 */
	@Synchronized
	fun predictDepth(input: Bitmap): DepthModelOutputs {
		// Frame ohne neue Bitmap auf Modellgröße skalieren
		srcRect.set(0, 0, input.width, input.height)
		scaledCanvas.drawBitmap(input, srcRect, dstRect, scalePaint)

		// Direkt in den vorhandenen Eingabe-Buffer schreiben
		NativeLib.bitmapToRgbHwc255FloatArray(scaledBitmap, inputBuffer.floatBuffer)

		outputIndex = (outputIndex + 1) % outputRing.size
		val outputs = outputRing[outputIndex]

		uniffi.NativeLib.runMetricDepthModelInferenceWithRaw(
			inputBuffer.asUniffiWrapper(),
			outputs.legacyMetricDepth.asUniffiWrapper(),
			outputs.rawRelativeDepth.asUniffiWrapper(),
		)

		return outputs
	}

	private companion object {
		const val OUTPUT_RING_SIZE = 4
	}
}