package com.algorithmic_alliance.eyeaiapp.inference

import android.graphics.Bitmap
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds

/** Configuration for the deliberately small scene-change representation. */
data class SceneChangeMonitorConfig(
	val sampleWidth: Int = 16,
	val sampleHeight: Int = 12,
	val sampleCadence: Duration = 100.milliseconds,
	val noiseFloor: Double = 8.0 / 255.0,
	val exposureCompensationLimit: Double = 32.0 / 255.0,
	val exposureConsistencyTolerance: Double = 16.0 / 255.0,
	val exposureConsistencyFraction: Double = 0.75,
) {
	init {
		require(sampleWidth > 0) { "sampleWidth must be positive" }
		require(sampleHeight > 0) { "sampleHeight must be positive" }
		require(sampleWidth.toLong() * sampleHeight <= Int.MAX_VALUE) {
			"sampleWidth * sampleHeight is too large"
		}
		require(sampleCadence.isFinite() && sampleCadence >= Duration.ZERO) {
			"sampleCadence must be finite and non-negative"
		}
		require(noiseFloor.isFinite() && noiseFloor >= 0.0 && noiseFloor < 1.0) {
			"noiseFloor must be finite and below 1.0"
		}
	require(exposureCompensationLimit.isFinite() && exposureCompensationLimit in 0.0..1.0) {
			"exposureCompensationLimit must be finite and in the range 0.0..1.0"
		}
		require(exposureConsistencyTolerance.isFinite() && exposureConsistencyTolerance in 0.0..1.0) {
			"exposureConsistencyTolerance must be finite and in the range 0.0..1.0"
		}
		require(exposureConsistencyFraction.isFinite() && exposureConsistencyFraction > 0.0) {
			"exposureConsistencyFraction must be finite and greater than zero"
		}
		require(exposureConsistencyFraction <= 1.0) {
			"exposureConsistencyFraction must not exceed 1.0"
		}
	}

	internal val sampleCount: Int
		get() = sampleWidth * sampleHeight

	internal val sampleCadenceNanos: Long
		get() = sampleCadence.inWholeNanoseconds
}

/** Result of one monitor call. A skipped call returns the last computed score. */
data class SceneChangeResult(
	val visualChangeScore: Double,
	val sampled: Boolean,
	val baselineFrame: Boolean,
	val sampledAtNanos: Long?,
) {
	/** Short alias useful at call sites that only need the numeric score. */
	val score: Double
		get() = visualChangeScore
}

/**
 * Source-neutral scene-change monitor with small Bitmap and luma-plane entry points.
 *
 * Each sampled frame is reduced to one luma value at the centre of each grid cell. The monitor
 * owns only the sample buffer and the scorer's small baseline; it never stores a Bitmap or the
 * caller's luma array. Public methods are serialized so CameraX, media, and future stream callers
 * can safely share one instance.
 */
class SceneChangeMonitor(
	val config: SceneChangeMonitorConfig = SceneChangeMonitorConfig(),
) {
	constructor(
		sampleWidth: Int,
		sampleHeight: Int,
		sampleCadence: Duration = 100.milliseconds,
		noiseFloor: Double = 8.0 / 255.0,
		exposureCompensationLimit: Double = 32.0 / 255.0,
		exposureConsistencyTolerance: Double = 16.0 / 255.0,
		exposureConsistencyFraction: Double = 0.75,
	) : this(
		SceneChangeMonitorConfig(
			sampleWidth = sampleWidth,
			sampleHeight = sampleHeight,
			sampleCadence = sampleCadence,
			noiseFloor = noiseFloor,
			exposureCompensationLimit = exposureCompensationLimit,
			exposureConsistencyTolerance = exposureConsistencyTolerance,
			exposureConsistencyFraction = exposureConsistencyFraction,
		)
	)

	private val sampleBuffer = ByteArray(config.sampleCount)
	private val scorer = LumaSceneChangeScorer(
		sampleCount = config.sampleCount,
		noiseFloor = config.noiseFloor,
		exposureCompensationLimit = config.exposureCompensationLimit,
		exposureConsistencyTolerance = config.exposureConsistencyTolerance,
		exposureConsistencyFraction = config.exposureConsistencyFraction,
	)

	private var sourceWidth = 0
	private var sourceHeight = 0
	private var sourceRotation = 0
	private var lastInputTimestampNanos: Long? = null
	private var lastSampleTimestampNanos: Long? = null
	private var scoreValue = 0.0
	private var sampledValue = false
	private var baselineFrameValue = false

	/** Last computed score, or 0.0 before the first baseline sample. */
	val lastScore: Double
		get() = synchronized(this) { scoreValue }

	/** Whether a baseline is currently available. */
	val hasBaseline: Boolean
		get() = synchronized(this) { scorer.hasBaseline }

	/** Whether the most recent call actually sampled a frame. */
	val lastCallWasSampled: Boolean
		get() = synchronized(this) { sampledValue }

	/** Whether the most recent sampled call only established a new baseline. */
	val lastCallWasBaselineFrame: Boolean
		get() = synchronized(this) { baselineFrameValue }

	/** Timestamp of the most recent sampled frame, if one exists. */
	val lastSampledAtNanos: Long?
		get() = synchronized(this) { lastSampleTimestampNanos }

	/**
	 * Samples an Android Bitmap and returns the current visual-change score.
	 *
	 * [rotationDegrees] is metadata for the source representation. A change in it starts a new
	 * baseline; the monitor deliberately does not keep or rotate the Bitmap.
	 */
	@Synchronized
	fun update(
		bitmap: Bitmap,
		rotationDegrees: Int = 0,
		timestampNanos: Long = System.nanoTime(),
	): Double = updateBitmapScore(bitmap, rotationDegrees, timestampNanos)

	/**
	 * Allocation-bearing diagnostic variant of [update] that also reports cadence and baseline
	 * state. The normal frame path should use [update] and the status properties instead.
	 */
	@Synchronized
	fun updateDetailed(
		bitmap: Bitmap,
		rotationDegrees: Int = 0,
		timestampNanos: Long = System.nanoTime(),
	): SceneChangeResult {
		updateBitmapScore(bitmap, rotationDegrees, timestampNanos)
		return result()
	}

	private fun updateBitmapScore(
		bitmap: Bitmap,
		rotationDegrees: Int,
		timestampNanos: Long,
	): Double {
		require(!bitmap.isRecycled) { "Cannot inspect a recycled Bitmap" }
		if (!prepareObservation(bitmap.width, bitmap.height, rotationDegrees, timestampNanos)) {
			return scoreValue
		}

		sampleBitmap(bitmap)
		return finishSample(timestampNanos)
	}

	/**
	 * Samples a tightly or strided 8-bit luma plane. This is the reusable path for non-Android
	 * sources; an eventual WebRTC I420 adapter can pass its Y plane here without changing the
	 * scoring algorithm.
	 */
	@Synchronized
	fun updateLuma(
		luma: ByteArray,
		width: Int,
		height: Int,
		rowStride: Int = width,
		rotationDegrees: Int = 0,
		timestampNanos: Long = System.nanoTime(),
	): Double = updateLumaScore(luma, width, height, rowStride, rotationDegrees, timestampNanos)

	/** Allocation-bearing diagnostic variant of [updateLuma]. */
	@Synchronized
	fun updateLumaDetailed(
		luma: ByteArray,
		width: Int,
		height: Int,
		rowStride: Int = width,
		rotationDegrees: Int = 0,
		timestampNanos: Long = System.nanoTime(),
	): SceneChangeResult {
		updateLumaScore(luma, width, height, rowStride, rotationDegrees, timestampNanos)
		return result()
	}

	private fun updateLumaScore(
		luma: ByteArray,
		width: Int,
		height: Int,
		rowStride: Int,
		rotationDegrees: Int,
		timestampNanos: Long,
	): Double {
		validateLumaPlane(luma, width, height, rowStride)
		if (!prepareObservation(width, height, rotationDegrees, timestampNanos)) {
			return scoreValue
		}

		sampleLuma(luma, width, height, rowStride)
		return finishSample(timestampNanos)
	}

	/** Clears the baseline and cadence history. The next frame is baseline-only. */
	@Synchronized
	fun reset() {
		scorer.reset()
		sourceWidth = 0
		sourceHeight = 0
		sourceRotation = 0
		lastInputTimestampNanos = null
		lastSampleTimestampNanos = null
		scoreValue = 0.0
		sampledValue = false
		baselineFrameValue = false
	}

	private fun prepareObservation(
		width: Int,
		height: Int,
		rotationDegrees: Int,
		timestampNanos: Long,
	): Boolean {
		val normalizedRotation = normalizeRotation(rotationDegrees)
		val sourceChanged = sourceWidth != 0 &&
			(sourceWidth != width || sourceHeight != height || sourceRotation != normalizedRotation)
		val timestampWentBackwards = lastInputTimestampNanos?.let { timestampNanos < it } == true

		if (sourceWidth == 0 || sourceChanged || timestampWentBackwards) {
			if (sourceChanged || timestampWentBackwards) {
				scorer.reset()
			}
			lastSampleTimestampNanos = null
			scoreValue = 0.0
		}

		sourceWidth = width
		sourceHeight = height
		sourceRotation = normalizedRotation
		lastInputTimestampNanos = timestampNanos
		sampledValue = false
		baselineFrameValue = false

		return lastSampleTimestampNanos == null ||
			config.sampleCadenceNanos == 0L ||
			hasElapsed(timestampNanos, lastSampleTimestampNanos!!, config.sampleCadenceNanos)
	}

	private fun finishSample(timestampNanos: Long): Double {
		val isBaseline = !scorer.hasBaseline
		scoreValue = scorer.score(sampleBuffer)
		lastSampleTimestampNanos = timestampNanos
		sampledValue = true
		baselineFrameValue = isBaseline
		return scoreValue
	}

	private fun result(): SceneChangeResult = SceneChangeResult(
		visualChangeScore = scoreValue,
		sampled = sampledValue,
		baselineFrame = baselineFrameValue,
		sampledAtNanos = lastSampleTimestampNanos,
	)

	private fun sampleBitmap(bitmap: Bitmap) {
		var targetIndex = 0
		for (sampleY in 0 until config.sampleHeight) {
			val sourceY = sourceCoordinate(sampleY, config.sampleHeight, bitmap.height)
			for (sampleX in 0 until config.sampleWidth) {
				val sourceX = sourceCoordinate(sampleX, config.sampleWidth, bitmap.width)
				sampleBuffer[targetIndex++] = argbToLuma(bitmap.getPixel(sourceX, sourceY))
			}
		}
	}

	private fun sampleLuma(
		luma: ByteArray,
		width: Int,
		height: Int,
		rowStride: Int,
	) {
		var targetIndex = 0
		for (sampleY in 0 until config.sampleHeight) {
			val sourceY = sourceCoordinate(sampleY, config.sampleHeight, height)
			val rowStart = sourceY * rowStride
			for (sampleX in 0 until config.sampleWidth) {
				val sourceX = sourceCoordinate(sampleX, config.sampleWidth, width)
				sampleBuffer[targetIndex++] = luma[rowStart + sourceX]
			}
		}
	}

	private fun validateLumaPlane(
		luma: ByteArray,
		width: Int,
		height: Int,
		rowStride: Int,
	) {
		require(width > 0 && height > 0) { "Luma dimensions must be positive" }
		require(rowStride >= width) { "rowStride must not be smaller than width" }
		val requiredBytes = (height.toLong() - 1L) * rowStride + width
		require(requiredBytes <= luma.size) {
			"Luma plane is too small for ${width}x$height with rowStride $rowStride"
		}
	}

	private fun hasElapsed(now: Long, then: Long, interval: Long): Boolean {
		if (now < then) return false
		return now - then >= interval
	}

	private fun normalizeRotation(rotationDegrees: Int): Int {
		val normalized = rotationDegrees % FULL_ROTATION_DEGREES
		return if (normalized < 0) normalized + FULL_ROTATION_DEGREES else normalized
	}

	private fun sourceCoordinate(sampleIndex: Int, sampleSize: Int, sourceSize: Int): Int =
		(((sampleIndex.toLong() * 2L + 1L) * sourceSize.toLong()) /
			(sampleSize.toLong() * 2L)).toInt().coerceIn(0, sourceSize - 1)

	private fun argbToLuma(color: Int): Byte {
		val red = (color shr 16) and 0xff
		val green = (color shr 8) and 0xff
		val blue = color and 0xff
		return ((77 * red + 150 * green + 29 * blue + 128) shr 8).toByte()
	}

	private companion object {
		const val FULL_ROTATION_DEGREES = 360
	}
}
