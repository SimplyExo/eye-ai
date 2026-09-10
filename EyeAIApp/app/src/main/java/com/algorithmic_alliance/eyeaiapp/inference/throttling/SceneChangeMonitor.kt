package com.algorithmic_alliance.eyeaiapp.inference.throttling

import android.graphics.Bitmap

data class SceneChangeConfig(
	val sampleWidth: Int = 16,
	val sampleHeight: Int = 12,
	val sampleCadenceNanos: Long = 100_000_000L,
	val noiseFloor: Double = 8.0 / 255.0,
	val exposureCompensationLimit: Double = 32.0 / 255.0,
	val exposureConsistencyTolerance: Double = 16.0 / 255.0,
	val exposureConsistencyFraction: Double = 0.75,
) {
	init {
		require(sampleWidth > 0 && sampleHeight > 0) { "Sample dimensions must be positive" }
		require(sampleWidth.toLong() * sampleHeight <= Int.MAX_VALUE) { "Sample grid is too large" }
		require(sampleCadenceNanos >= 0L) { "sampleCadenceNanos must be non-negative" }
	}

	val sampleCount: Int
		get() = sampleWidth * sampleHeight
}

data class SceneSample(
	val score: Double,
	val sampledAtNanos: Long,
	val baselineFrame: Boolean,
)

class SceneChangeMonitor(
	val config: SceneChangeConfig = SceneChangeConfig(),
) {
	private val sampleBuffer = ByteArray(config.sampleCount)
	private val scorer = LumaSceneChangeScorer(
		config.sampleCount,
		config.noiseFloor,
		config.exposureCompensationLimit,
		config.exposureConsistencyTolerance,
		config.exposureConsistencyFraction,
	)
	private var sourceWidth = 0
	private var sourceHeight = 0
	private var sourceRotation = 0
	private var lastInputNanos: Long? = null
	private var lastSampleNanos: Long? = null

	fun sample(bitmap: Bitmap, rotationDegrees: Int, nowNanos: Long): SceneSample? {
		require(!bitmap.isRecycled) { "Cannot inspect a recycled Bitmap" }
		if (!prepare(bitmap.width, bitmap.height, rotationDegrees, nowNanos)) return null
		var target = 0
		for (sampleY in 0 until config.sampleHeight) {
			val sourceY = sourceCoordinate(sampleY, config.sampleHeight, bitmap.height)
			for (sampleX in 0 until config.sampleWidth) {
				val sourceX = sourceCoordinate(sampleX, config.sampleWidth, bitmap.width)
				sampleBuffer[target++] = argbToLuma(bitmap.getPixel(sourceX, sourceY))
			}
		}
		return finish(nowNanos)
	}

	fun sampleLuma(
		luma: ByteArray,
		width: Int,
		height: Int,
		rowStride: Int = width,
		rotationDegrees: Int = 0,
		nowNanos: Long,
	): SceneSample? {
		validateLuma(luma, width, height, rowStride)
		if (!prepare(width, height, rotationDegrees, nowNanos)) return null
		var target = 0
		for (sampleY in 0 until config.sampleHeight) {
			val sourceY = sourceCoordinate(sampleY, config.sampleHeight, height)
			val rowStart = sourceY * rowStride
			for (sampleX in 0 until config.sampleWidth) {
				val sourceX = sourceCoordinate(sampleX, config.sampleWidth, width)
				sampleBuffer[target++] = luma[rowStart + sourceX]
			}
		}
		return finish(nowNanos)
	}

	fun reset() {
		scorer.reset()
		sourceWidth = 0
		sourceHeight = 0
		sourceRotation = 0
		lastInputNanos = null
		lastSampleNanos = null
	}

	private fun prepare(width: Int, height: Int, rotationDegrees: Int, now: Long): Boolean {
		val rotation = Math.floorMod(rotationDegrees, 360)
		val changed = sourceWidth != 0 &&
			(sourceWidth != width || sourceHeight != height || sourceRotation != rotation)
		val timeWentBackwards = lastInputNanos?.let { now < it } == true
		if (changed || timeWentBackwards) reset()

		sourceWidth = width
		sourceHeight = height
		sourceRotation = rotation
		lastInputNanos = now
		return lastSampleNanos?.let { now - it >= config.sampleCadenceNanos } ?: true
	}

	private fun finish(now: Long): SceneSample {
		val baseline = !scorer.hasBaseline
		val score = scorer.score(sampleBuffer)
		lastSampleNanos = now
		return SceneSample(score, now, baseline)
	}

	private fun validateLuma(luma: ByteArray, width: Int, height: Int, rowStride: Int) {
		require(width > 0 && height > 0) { "Luma dimensions must be positive" }
		require(rowStride >= width) { "rowStride must not be smaller than width" }
		val required = (height.toLong() - 1L) * rowStride + width
		require(required <= luma.size) { "Luma plane is too small for the supplied dimensions" }
	}

	private fun sourceCoordinate(sampleIndex: Int, sampleSize: Int, sourceSize: Int): Int =
		(((sampleIndex.toLong() * 2L + 1L) * sourceSize) / (sampleSize.toLong() * 2L))
			.toInt()
			.coerceIn(0, sourceSize - 1)

	private fun argbToLuma(color: Int): Byte {
		val red = (color shr 16) and 0xff
		val green = (color shr 8) and 0xff
		val blue = color and 0xff
		return ((77 * red + 150 * green + 29 * blue + 128) shr 8).toByte()
	}
}
