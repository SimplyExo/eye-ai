package com.algorithmic_alliance.eyeaiapp

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.media.Image
import android.util.Log
import android.util.Size
import androidx.core.graphics.createBitmap
import uniffi.NativeLib.UniffiDetectedObject
import uniffi.NativeLib.UniffiFloatBufferWrapper
import uniffi.NativeLib.UniffiIntBufferWrapper
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
		val capacity: Int get() = floatBuffer.capacity()

		fun asUniffiWrapper(): UniffiFloatBufferWrapper {
			return UniffiFloatBufferWrapper(
				getByteBufferPtr(byteBuffer),
				floatBuffer.capacity()
			)
		}

		fun rewind() {
			byteBuffer.rewind()
			floatBuffer.rewind()
		}
	}

	class NativeIntBuffer(length: Int) {
		var byteBuffer = ByteBuffer
			.allocateDirect(length * Int.SIZE_BYTES)
			.order(ByteOrder.nativeOrder())

		var intBuffer = byteBuffer.asIntBuffer()
		val capacity: Int get() = intBuffer.capacity()

		fun asUniffiWrapper(): UniffiIntBufferWrapper {
			return UniffiIntBufferWrapper(
				getByteBufferPtr(byteBuffer),
				intBuffer.capacity()
			)
		}

		fun rewind() {
			byteBuffer.rewind()
			intBuffer.rewind()
		}
	}


	external fun bitmapToRgbHwc255FloatArray(
		bitmap: Bitmap,
		outFloatBuffer: FloatBuffer
	)


	fun bitmapToRgbHwc255FloatArray(
		bitmap: Bitmap,
		reuseBuffer: NativeFloatBuffer? = null
	): NativeFloatBuffer {
		val size = bitmap.width * bitmap.height * 3
		val floatBuffer = reuseBuffer?.takeIf { it.capacity >= size }
			?: NativeFloatBuffer(size)
		floatBuffer.rewind()
		bitmapToRgbHwc255FloatArray(bitmap, floatBuffer.floatBuffer)
		return floatBuffer
	}


	/** @param input values should be between 0.0f and 1.0f */
	fun metricDepthColormap(
		input: UniffiFloatBufferWrapper,
		inputImageSize: Size,
		reuseIntBuffer: NativeIntBuffer? = null,
		reuseIntArray: IntArray? = null,
		reuseBitmap: Bitmap? = null
	): Bitmap {
		if (input.length != inputImageSize.width * inputImageSize.height) {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"input depth array length ${input.length} does not match output bitmap size $inputImageSize"
			)
			return reuseBitmap?.takeIf {
				it.width == inputImageSize.width && it.height == inputImageSize.height
			} ?: createBitmap(inputImageSize.width, inputImageSize.height)
		}

		val colormappedPixels = reuseIntBuffer?.takeIf { it.capacity >= input.length }
			?: NativeIntBuffer(input.length)
		colormappedPixels.rewind()

		uniffi.NativeLib.metricDepthColormap(
			input,
			colormappedPixels.asUniffiWrapper()
		)

		colormappedPixels.intBuffer.rewind()
		val colormappedPixelsArray = reuseIntArray?.takeIf { it.size >= input.length }
			?: IntArray(colormappedPixels.capacity)
		colormappedPixels.intBuffer.get(colormappedPixelsArray, 0, input.length)

		val result = reuseBitmap?.takeIf { it.isMutable }
			?: createBitmap(inputImageSize.width, inputImageSize.height)
		result.setPixels(
			colormappedPixelsArray, 0, inputImageSize.width,
			0, 0, inputImageSize.width, inputImageSize.height
		)
		return result
	}


	fun copyImagePixels(image: Image, destBitmap: Bitmap) {
		require(image.format == PixelFormat.RGBA_8888) {
			"Unsupported image format: ${image.format}. Expected RGBA_8888"
		}
		require(destBitmap.width == image.width && destBitmap.height == image.height) {
			"destBitmap ${destBitmap.width}x${destBitmap.height} != image ${image.width}x${image.height}"
		}

		val plane = image.planes[0]
		val buffer = plane.buffer
		val pixelStride = plane.pixelStride
		val rowStride = plane.rowStride
		val width = image.width
		val height = image.height

		if (pixelStride == 4 && rowStride == width * 4) {
			buffer.rewind()
			destBitmap.copyPixelsFromBuffer(buffer)
		} else {
			val rgbaBytes = ByteArray(width * height * 4)
			for (row in 0 until height) {
				val startPos = row * rowStride
				buffer.position(startPos)
				buffer.get(rgbaBytes, row * width * 4, width * 4)
			}
			destBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(rgbaBytes))
		}
	}


	fun imageToBitmap(image: Image, rotationDegrees: Float): Bitmap {
		val width = image.width
		val height = image.height
		val rawBitmap = createBitmap(width, height)
		copyImagePixels(image, rawBitmap)
		return rotateBitmap(rawBitmap, rotationDegrees)
	}


	fun rotateBitmap(
		bitmap: Bitmap,
		rotationDegrees: Float,
		reuseBitmap: Bitmap? = null
	): Bitmap {
		if (rotationDegrees == 0f) return bitmap

		val rotatedW: Int
		val rotatedH: Int
		if (rotationDegrees == 90f || rotationDegrees == 270f) {
			rotatedW = bitmap.height
			rotatedH = bitmap.width
		} else {
			rotatedW = bitmap.width
			rotatedH = bitmap.height
		}

		val result = reuseBitmap?.takeIf {
			it.width == rotatedW && it.height == rotatedH && it.isMutable
		} ?: createBitmap(rotatedW, rotatedH)

		val matrix = Matrix().apply { postRotate(rotationDegrees) }
		val rect = android.graphics.RectF(
			0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat()
		)
		matrix.mapRect(rect)
		matrix.postTranslate(-rect.left, -rect.top)
		Canvas(result).drawBitmap(bitmap, matrix, null)
		return result
	}
}