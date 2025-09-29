package com.algorithmic_alliance.eyeaiapp

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.media.Image
import android.util.Log
import android.util.Size
import androidx.core.graphics.createBitmap
import java.nio.ByteBuffer

// see NativeLib.cpp
enum class ProfilingFrameType(val id: Int) {
	Depth(0), Object(1)
}

/** Kotlin interface with NativeLib c++ code */
object NativeLib {
	init {
		System.loadLibrary("NativeLib")
	}

	// Yolo
	external fun initYoloRuntime(
		model: ByteArray,
		labels: Array<String>,
		gpuDelegateSerializationDir: String,
		modelToken: String,
		enableNpu: Boolean,
		skelDirectory: String
	): Boolean

	external fun runYoloOperation(input: FloatArray): String

	external fun getYoloInputShape(): IntArray

	external fun getYoloOutputShape(): IntArray

	external fun newDepthFrame()
	external fun formatDepthFrame(): String
	external fun newCameraFrame()
	external fun formatCameraFrame(): String
	external fun newObjectFrame()
	external fun formatObjectFrame(): String

	external fun initMetricDepthModel(
		relativeDepthModel: ByteArray,
		gpuDelegateSerializationDir: String,
		relativeDepthModelToken: String,
		enableNpu: Boolean,
		skelDirectory: String
	)

	external fun shutdownMetricDepthModel()

	external fun runMetricDepthModelInference(
		input: FloatArray,
		output: FloatArray
	)

	external fun getMetricDepthModelInputShape(): IntArray

	external fun getMetricDepthModelOutputShape(): IntArray

	external fun metricDepthColormap(depthValues: FloatArray, colormappedPixels: IntArray)

	external fun bitmapToRgbHwc255FloatArray(
		bitmap: Bitmap,
		outFloatArray: FloatArray,
		profilingFrameType: Int
	)

	external fun setupAudioSettings(cocoLabelsAudio: ByteArray, cocoLabelsData: ByteArray)
	external fun setAudioSettings(frequency: Int, incidence: Int)
	external fun sendAIData(array: FloatArray)
	external fun setDepthAudioPaused(paused: Boolean)
	external fun setObjectAudioPaused(paused: Boolean)
	external fun getProcessingStatus(): Boolean
	external fun destroySpatialAudio()

	//external fun playSound(frequency: Float, duration: Float)

	/** @param input values should be between 0.0f and 1.0f */
	fun metricDepthColormap(input: FloatArray, inputImageSize: Size): Bitmap {
		if (input.size != inputImageSize.width * inputImageSize.height) {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"input depth array length does not match output bitmap size"
			)
			return createBitmap(inputImageSize.width, inputImageSize.height)
		}

		val colormappedPixels = IntArray(input.size)

		metricDepthColormap(input, colormappedPixels)

		return Bitmap.createBitmap(
			colormappedPixels,
			inputImageSize.width,
			inputImageSize.height,
			Bitmap.Config.ARGB_8888
		)
	}

	fun bitmapToRgbHwc255FloatArray(
		bitmap: Bitmap,
		profilingFrameType: ProfilingFrameType
	): FloatArray {
		val floatArray = FloatArray(bitmap.width * bitmap.height * 3)

		bitmapToRgbHwc255FloatArray(bitmap, floatArray, profilingFrameType.id)

		return floatArray
	}

	fun imageToBitmap(image: Image, rotationDegrees: Float): Bitmap {
		require(image.format == PixelFormat.RGBA_8888) {
			"Unsupported image format: ${image.format}. Expected RGBA_8888"
		}

		val plane = image.planes[0]
		val buffer = plane.buffer
		val pixelStride = plane.pixelStride
		val rowStride = plane.rowStride

		val width = image.width
		val height = image.height

		val bitmap = createBitmap(width, height)

		// If there's no padding between rows, we can do a direct copy
		if (pixelStride == 4 && rowStride == width * 4) {
			bitmap.copyPixelsFromBuffer(buffer)
		} else {
			// Handle cases with padding between rows
			val rgbaBytes = ByteArray(width * height * 4)
			for (row in 0 until height) {
				val startPos = row * rowStride
				buffer.position(startPos)
				buffer.get(rgbaBytes, row * width * 4, width * 4)
			}
			bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(rgbaBytes))
		}

		return rotateBitmap(bitmap, rotationDegrees)
	}

	fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Float): Bitmap =
		Bitmap.createBitmap(
			bitmap,
			0,
			0,
			bitmap.width,
			bitmap.height,
			Matrix().apply { postRotate(rotationDegrees) },
			false
		)
}
