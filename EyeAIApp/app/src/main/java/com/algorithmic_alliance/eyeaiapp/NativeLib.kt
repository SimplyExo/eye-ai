package com.algorithmic_alliance.eyeaiapp

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.media.Image
import android.os.Build
import android.util.Log
import android.util.Size
import androidx.core.graphics.createBitmap
import uniffi.NativeLib.UniffiDetectedObject
import uniffi.NativeLib.UniffiFloatBufferWrapper
import uniffi.NativeLib.UniffiIntBufferWrapper
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer


/** Kotlin interface with NativeLib c++ code */
object NativeLib {
	init {
		System.loadLibrary("NativeLib")
	}

	// uniffi Float-/ByteArray zero-copy helper functions:
	external fun getByteBufferPtr(buffer: ByteBuffer): Long

	class NativeFloatBuffer(length: Int) {
		var byteBuffer = ByteBuffer
			.allocateDirect(length * Float.SIZE_BYTES)
			.order(ByteOrder.nativeOrder())

		var floatBuffer = byteBuffer.asFloatBuffer()

		fun asUniffiWrapper(): UniffiFloatBufferWrapper {
			return UniffiFloatBufferWrapper(
				getByteBufferPtr(byteBuffer),
				floatBuffer.capacity()
			)
		}
	}

	class NativeIntBuffer(length: Int) {
		var byteBuffer = ByteBuffer
			.allocateDirect(length * Int.SIZE_BYTES)
			.order(ByteOrder.nativeOrder())

		var intBuffer = byteBuffer.asIntBuffer()

		fun asUniffiWrapper(): UniffiIntBufferWrapper {
			return UniffiIntBufferWrapper(
				getByteBufferPtr(byteBuffer),
				intBuffer.capacity()
			)
		}
	}


	external fun bitmapToRgbHwc255FloatArray(
		bitmap: Bitmap,
		outFloatBuffer: FloatBuffer
	)


	/** @param input values should be between 0.0f and 1.0f */
	fun metricDepthColormap(input: UniffiFloatBufferWrapper, inputImageSize: Size): Bitmap {
		if (input.length != inputImageSize.width * inputImageSize.height) {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"input depth array length ${input.length} does not match output bitmap size $inputImageSize"
			)
			return createBitmap(inputImageSize.width, inputImageSize.height)
		}

		val colormappedPixels = NativeIntBuffer(input.length)

		uniffi.NativeLib.metricDepthColormap(
			input,
			colormappedPixels.asUniffiWrapper()
		)
		//metricDepthColormap(input, colormappedPixels)

		// TODO: improve Bitmap/Buffer/Array conversions...
		val colormappedPixelsArray = IntArray(colormappedPixels.intBuffer.remaining())
		colormappedPixels.intBuffer.get(colormappedPixelsArray)

		return Bitmap.createBitmap(
			colormappedPixelsArray,
			inputImageSize.width,
			inputImageSize.height,
			Bitmap.Config.ARGB_8888
		)
	}

	fun bitmapToRgbHwc255FloatArray(
		bitmap: Bitmap
	): NativeFloatBuffer {
		val floatBuffer = NativeFloatBuffer(bitmap.width * bitmap.height * 3)

		bitmapToRgbHwc255FloatArray(bitmap, floatBuffer.floatBuffer)

		return floatBuffer
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

	fun createSerializedDelegateCacheDirectory(context: Context): File {
		val gpuDelegateCacheDirectory = File(context.cacheDir, "gpu_delegate_cache")
		if (!gpuDelegateCacheDirectory.exists()) gpuDelegateCacheDirectory.mkdirs()
		return gpuDelegateCacheDirectory
	}

	/**
	 * generates a unique token based on the model file name and last install/update time of this app
	 */
	fun getModelToken(context: Context, modelFilename: String): String {
		val lastUpdateTime = getLastAppUpdateTime(context)
		return "${modelFilename}_${lastUpdateTime}"
	}
}