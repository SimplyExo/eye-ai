package com.algorithmic_alliance.eyeaiapp.inference.throttling

import kotlin.math.abs

class LumaSceneChangeScorer(
	sampleCount: Int,
	noiseFloor: Double = 8.0 / 255.0,
	exposureCompensationLimit: Double = 32.0 / 255.0,
	exposureConsistencyTolerance: Double = 16.0 / 255.0,
	private val exposureConsistencyFraction: Double = 0.75,
) {
	private val count = sampleCount.also { require(it > 0) { "sampleCount must be positive" } }
	private val baseline = ByteArray(count)
	private var baselineAvailable = false
	private val noiseFloorLuma = fraction(noiseFloor, "noiseFloor") * LUMA_MAX
	private val exposureCompensationLimitLuma =
		fraction(exposureCompensationLimit, "exposureCompensationLimit") * LUMA_MAX
	private val exposureConsistencyToleranceLuma =
		fraction(exposureConsistencyTolerance, "exposureConsistencyTolerance") * LUMA_MAX

	init {
		require(noiseFloor < 1.0) { "noiseFloor must be below 1.0" }
		require(exposureConsistencyFraction.isFinite() && exposureConsistencyFraction in 0.0..1.0) {
			"exposureConsistencyFraction must be finite and in the range 0.0..1.0"
		}
		require(exposureConsistencyFraction > 0.0) {
			"exposureConsistencyFraction must be greater than zero"
		}
	}

	val hasBaseline: Boolean
		get() = baselineAvailable

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
			signedDeltaSum += (unsigned(currentLuma[index]) - unsigned(baseline[index])).toLong()
		}
		val meanDelta = signedDeltaSum.toDouble() / count
		val globalOffset = if (isConsistentExposureShift(currentLuma, meanDelta)) meanDelta else 0.0

		var effectiveDifferenceSum = 0.0
		for (index in 0 until count) {
			val delta = unsigned(currentLuma[index]) - unsigned(baseline[index])
			effectiveDifferenceSum +=
				(abs(delta - globalOffset) - noiseFloorLuma).coerceAtLeast(0.0)
		}
		currentLuma.copyInto(baseline)
		return (effectiveDifferenceSum / count / (LUMA_MAX - noiseFloorLuma)).coerceIn(0.0, 1.0)
	}

	fun reset() {
		baselineAvailable = false
	}

	private fun isConsistentExposureShift(currentLuma: ByteArray, meanDelta: Double): Boolean {
		if (abs(meanDelta) > exposureCompensationLimitLuma) return false
		var consistent = 0
		for (index in 0 until count) {
			val delta = unsigned(currentLuma[index]) - unsigned(baseline[index])
			if (abs(delta - meanDelta) <= exposureConsistencyToleranceLuma) consistent++
		}
		return consistent.toDouble() / count >= exposureConsistencyFraction
	}

	private fun unsigned(value: Byte): Int = value.toInt() and 0xff

	private companion object {
		const val LUMA_MAX = 255.0

		fun fraction(value: Double, name: String): Double {
			require(value.isFinite() && value in 0.0..1.0) {
				"$name must be finite and in the range 0.0..1.0"
			}
			return value
		}
	}
}
