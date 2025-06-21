package com.algorithmic_alliance.eyeaiapp

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.media.Image
import android.util.Log
import android.util.Size
import androidx.core.graphics.createBitmap

/** Kotlin interface with NativeLib c++ code */
object NativeLib {
	init {
		System.loadLibrary("NativeLib")
	}

	external fun newDepthFrame()
	external fun formatDepthFrame(): String
	external fun newCameraFrame()
	external fun formatCameraFrame(): String

	external fun initDepthTfLiteRuntime(
		model: ByteArray,
		gpuDelegateSerializationDir: String,
		modelToken: String
	)

	external fun shutdownDepthTfLiteRuntime()

	external fun runDepthTfLiteInference(
		input: FloatArray,
		output: FloatArray,
		meanR: Float,
		meanG: Float,
		meanB: Float,
		stddevR: Float,
		stddevG: Float,
		stddevB: Float
	)

	external fun depthColormap(depthValues: FloatArray, colormappedPixels: IntArray)

	external fun bitmapToRgbChwFloatArray(bitmap: Bitmap, outFloatArray: FloatArray)

	external fun bitmapToRgbHwc255FloatArray(bitmap: Bitmap, outFloatArray: FloatArray)

	external fun imageBytesToArgbIntArray(imageBytes: ByteArray, outIntArray: IntArray)

	/** @param input values should be between 0.0f and 1.0f */
	fun depthColorMap(input: FloatArray, inputImageSize: Size): Bitmap {
		if (input.size != inputImageSize.width * inputImageSize.height) {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"input depth array length does not match output bitmap size"
			)
			return createBitmap(inputImageSize.width, inputImageSize.height)
		}

		val colormappedPixels = IntArray(input.size)

		depthColormap(input, colormappedPixels)

		return Bitmap.createBitmap(
			colormappedPixels,
			inputImageSize.width,
			inputImageSize.height,
			Bitmap.Config.ARGB_8888
		)
	}

	fun bitmapToRgbChwFloatArray(bitmap: Bitmap): FloatArray {
		val floatArray = FloatArray(bitmap.width * bitmap.height * 3)

		bitmapToRgbChwFloatArray(bitmap, floatArray)

		return floatArray
	}

	fun bitmapToRgbHwc255FloatArray(bitmap: Bitmap): FloatArray {
		val floatArray = FloatArray(bitmap.width * bitmap.height * 3)

		bitmapToRgbHwc255FloatArray(bitmap, floatArray)

		return floatArray
	}

	fun imageToBitmap(image: Image, rotationDegrees: Float): Bitmap {
		require(image.format == PixelFormat.RGBA_8888)

		val pixelBuffer = image.planes[0].buffer
		val pixelBytes = ByteArray(pixelBuffer.remaining())
		pixelBuffer.get(pixelBytes)
		require(pixelBytes.size == image.width * image.height * 4)

		val pixels = IntArray(image.width * image.height)

		imageBytesToArgbIntArray(pixelBytes, pixels)

		return rotateBitmap(
			Bitmap.createBitmap(
				pixels,
				image.width,
				image.height,
				Bitmap.Config.ARGB_8888
			),
			rotationDegrees
		)
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
