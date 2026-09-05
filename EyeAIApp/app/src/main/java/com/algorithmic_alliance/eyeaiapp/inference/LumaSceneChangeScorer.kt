package com.algorithmic_alliance.eyeaiapp.inference

import kotlin.math.abs

/**
 * Pure Kotlin scorer for two small, equally sized 8-bit luma representations.
 *
 * The input array is used only while [score] runs. The scorer copies its values into a compact
 * baseline buffer and therefore never retains a source frame or the caller's array.
 */
class LumaSceneChangeScorer(
	sampleCount: Int,
	private val noiseFloor: Double = DEFAULT_NOISE_FLOOR,
	private val exposureCompensationLimit: Double = DEFAULT_EXPOSURE_COMPENSATION_LIMIT,
	private val exposureConsistencyTolerance: Double = DEFAULT_EXPOSURE_CONSISTENCY_TOLERANCE,
	private val exposureConsistencyFraction: Double = DEFAULT_EXPOSURE_CONSISTENCY_FRACTION,
) {
	private val count = requireSampleCount(sampleCount)
	private val baseline = ByteArray(count)
	private var baselineAvailable = false

	private val noiseFloorLuma = requireFraction(noiseFloor, "noiseFloor") * LUMA_MAX
	private val exposureCompensationLimitLuma =
		requireFraction(exposureCompensationLimit, "exposureCompensationLimit") * LUMA_MAX
	private val exposureConsistencyToleranceLuma =
		requireFraction(exposureConsistencyTolerance, "exposureConsistencyTolerance") * LUMA_MAX

	init {
		require(noiseFloor < 1.0) { "noiseFloor must be below 1.0" }
		require(exposureConsistencyFraction.isFinite() && exposureConsistencyFraction > 0.0) {
			"exposureConsistencyFraction must be finite and greater than zero"
		}
		require(exposureConsistencyFraction <= 1.0) {
			"exposureConsistencyFraction must not exceed 1.0"
		}
	}

	/** Whether a previous sample exists and can be compared against. */
	val hasBaseline: Boolean
		get() = synchronized(this) { baselineAvailable }

	/**
	 * Scores [currentLuma] and makes it the next baseline.
	 *
	 * Values are interpreted as unsigned bytes in the range 0..255. The first call only establishes
	 * the baseline and returns 0.0. The result is always clamped to 0.0..1.0.
	 */
	@Synchronized
	fun score(currentLuma: ByteArray): Double {
		require(currentLuma.size == count) {
			"Expected $count luma samples, got ${currentLuma.size}"
		}

		if (!baselineAvailable) {
			currentLuma.copyInto(baseline)
			baselineAvailable = true
			return 0.0
		}

		var signedDeltaSum = 0L
		for (index in 0 until count) {
			val delta = unsigned(currentLuma[index]) - unsigned(baseline[index])
			signedDeltaSum += delta.toLong()
		}

		val meanSignedDelta = signedDeltaSum.toDouble() / count
		val globalOffset = if (hasConsistentSmallGlobalShift(currentLuma, meanSignedDelta)) {
			meanSignedDelta
		} else {
			0.0
		}

		var effectiveDifferenceSum = 0.0
		for (index in 0 until count) {
			val delta = unsigned(currentLuma[index]) - unsigned(baseline[index])
			val difference = abs(delta.toDouble() - globalOffset)
			effectiveDifferenceSum += (difference - noiseFloorLuma).coerceAtLeast(0.0)
		}

		currentLuma.copyInto(baseline)

		val maximumEffectiveDifference = LUMA_MAX - noiseFloorLuma
		return (effectiveDifferenceSum / count / maximumEffectiveDifference).coerceIn(0.0, 1.0)
	}

	/** Clears the baseline without retaining or exposing any caller-owned frame data. */
	@Synchronized
	fun reset() {
		baselineAvailable = false
	}

	private fun hasConsistentSmallGlobalShift(
		currentLuma: ByteArray,
		meanSignedDelta: Double,
	): Boolean {
		if (abs(meanSignedDelta) > exposureCompensationLimitLuma) return false

		var consistentSamples = 0
		for (index in 0 until count) {
			val delta = unsigned(currentLuma[index]) - unsigned(baseline[index])
			if (abs(delta.toDouble() - meanSignedDelta) <= exposureConsistencyToleranceLuma) {
				consistentSamples++
			}
		}
		return consistentSamples.toDouble() / count >= exposureConsistencyFraction
	}

	private fun unsigned(value: Byte): Int = value.toInt() and 0xff

	private companion object {
		const val LUMA_MAX = 255.0
		const val DEFAULT_NOISE_FLOOR = 8.0 / 255.0
		const val DEFAULT_EXPOSURE_COMPENSATION_LIMIT = 32.0 / 255.0
		const val DEFAULT_EXPOSURE_CONSISTENCY_TOLERANCE = 16.0 / 255.0
		const val DEFAULT_EXPOSURE_CONSISTENCY_FRACTION = 0.75

		fun requireSampleCount(value: Int): Int {
			require(value > 0) { "sampleCount must be positive" }
			return value
		}

		fun requireFraction(value: Double, name: String): Double {
			require(value.isFinite() && value in 0.0..1.0) {
				"$name must be finite and in the range 0.0..1.0"
			}
			return value
		}
	}
}
