package com.algorithmic_alliance.eyeaiapp

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.media.Image
import android.util.Log
import android.util.Size
import androidx.core.graphics.createBitmap
import uniffi.NativeLib.UniffiFloatBufferWrapper
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.ByteOrder

// see NativeLib.cpp
enum class ProfilingFrameType(val id: Int) {
	Depth(0), Object(1)
}

/** Kotlin interface with NativeLib c++ code */
object NativeLib {
	init {
		System.loadLibrary("NativeLib")
	}

	val logger = object : uniffi.NativeLib.LogCallbacks {
		override fun logInfoCallback(msg: String) {
			Log.i("eye-ai-core-rs", msg)
		}

		override fun logWarningCallback(msg: String) {
			Log.w("eye-ai-core-rs", msg)
		}

		override fun logErrorCallback(msg: String) {
			Log.e("eye-ai-core-rs", msg)
		}
	}

	// Depth (wrapper functions to provide the logging interface to rust)
	fun initMetricDepthModel(
		relativeDepthModel: ByteArray,
		gpuDelegateSerializationDir: String,
		relativeDepthModelToken: String,
		enableNpu: Boolean,
		skelDirectory: String
	) {
		return uniffi.NativeLib.initMetricDepthModel(
			relativeDepthModel,
			gpuDelegateSerializationDir,
			relativeDepthModelToken,
			enableNpu,
			skelDirectory,
			logger
		)
	}

	fun runMetricDepthModelInference(input: UniffiFloatBufferWrapper, output: UniffiFloatBufferWrapper) {
		return uniffi.NativeLib.runMetricDepthModelInference(input, output, logger)
	}

	fun getMetricDepthModelInputShape(): List<Int> {
		return uniffi.NativeLib.getMetricDepthModelInputShape(logger)
	}

	fun getMetricDepthModelOutputShape(): List<Int> {
		return uniffi.NativeLib.getMetricDepthModelOutputShape(logger)
	}

	// Yolo (wrapper functions to provide the logging interface to rust)
	fun initYoloRuntime(
		model: ByteArray,
		labels: List<String>,
		gpuDelegateSerializationDir: String,
		modelToken: String,
		enableNpu: Boolean,
		skelDirectory: String
	) {
		return uniffi.NativeLib.initYoloRuntime(
			model,
			labels,
			gpuDelegateSerializationDir,
			modelToken,
			enableNpu,
			skelDirectory,
			logger
		)
	}

	fun runYoloOperation(input: UniffiFloatBufferWrapper): List<uniffi.NativeLib.UniffiDetectedObject> {
		return uniffi.NativeLib.runYoloOperation(input, logger)
	}

	fun getYoloInputShape(): List<Int> = uniffi.NativeLib.getYoloInputShape(logger)

	fun getYoloOutputShape(): List<Int> = uniffi.NativeLib.getYoloOutputShape(logger)

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


	external fun metricDepthColormap(depthValues: FloatBuffer, colormappedPixels: IntArray)

	external fun bitmapToRgbHwc255FloatArray(
		bitmap: Bitmap,
		outFloatBuffer: FloatBuffer,
		profilingFrameType: Int
	)

	external fun setupAudioSettings(cocoLabelsAudio: ByteArray, cocoLabelsData: ByteArray)
	external fun setAudioSettings(frequency: Int, incidence: Int)
	external fun sendAIData(array: FloatBuffer)
	external fun setDepthAudioPaused(paused: Boolean)
	external fun setObjectAudioPaused(paused: Boolean)
	external fun getProcessingStatus(): Boolean
	external fun destroySpatialAudio()

	//external fun playSound(frequency: Float, duration: Float)

	/** @param input values should be between 0.0f and 1.0f */
	fun metricDepthColormap(input: FloatBuffer, inputImageSize: Size): Bitmap {
		if (input.capacity() != inputImageSize.width * inputImageSize.height) {
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"input depth array length does not match output bitmap size"
			)
			return createBitmap(inputImageSize.width, inputImageSize.height)
		}

		val colormappedPixels = IntArray(input.capacity())

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
	): NativeFloatBuffer {
		val floatBuffer = NativeFloatBuffer(bitmap.width * bitmap.height * 3)

		bitmapToRgbHwc255FloatArray(bitmap, floatBuffer.floatBuffer, profilingFrameType.id)

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
}
