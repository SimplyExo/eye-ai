package com.algorithmic_alliance.eyeaiapp.inference

import kotlin.math.sqrt
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds

/** Technical, deliberately uncalibrated defaults for the phone-motion hint. */
data class PhoneMotionMonitorConfig(
	val staleTimeout: Duration = 500.milliseconds,
	val smoothingFactor: Double = 0.25,
	val gyroFullScaleRadPerSecond: Double = 2.0,
	val linearAccelerationFullScaleMetersPerSecondSquared: Double = 4.0,
	val linearAccelerationWeight: Double = 0.25,
) {
	init {
		require(staleTimeout.isFinite() && staleTimeout > Duration.ZERO) {
			"staleTimeout must be finite and greater than zero"
		}
		require(smoothingFactor.isFinite() && smoothingFactor > 0.0 && smoothingFactor <= 1.0) {
			"smoothingFactor must be finite and greater than zero and at most one"
		}
		require(gyroFullScaleRadPerSecond.isFinite() && gyroFullScaleRadPerSecond > 0.0) {
			"gyroFullScaleRadPerSecond must be finite and greater than zero"
		}
		require(
			linearAccelerationFullScaleMetersPerSecondSquared.isFinite() &&
				linearAccelerationFullScaleMetersPerSecondSquared > 0.0
		) {
			"linearAccelerationFullScaleMetersPerSecondSquared must be finite and greater than zero"
		}
		require(linearAccelerationWeight.isFinite() && linearAccelerationWeight in 0.0..1.0) {
			"linearAccelerationWeight must be finite and in the range 0.0..1.0"
		}
	}

	internal val staleTimeoutNanos: Long
		get() = staleTimeout.inWholeNanoseconds
}

/** Monotone time source used only for polling timeout validity. */
fun interface PhoneMotionClock {
	fun nowNanos(): Long
}

/** Small pure math boundary shared by Android and future motion sources. */
object PhoneMotionMath {
	/** Returns the Euclidean magnitude, or NaN if one input is not finite. */
	fun magnitude(x: Float, y: Float, z: Float): Double {
		if (!x.isFinite() || !y.isFinite() || !z.isFinite()) return Double.NaN
		val xDouble = x.toDouble()
		val yDouble = y.toDouble()
		val zDouble = z.toDouble()
		return sqrt(xDouble * xDouble + yDouble * yDouble + zDouble * zDouble)
	}

	/** Maps a non-negative magnitude to 0.0..1.0 and clamps sensor outliers. */
	fun normalizeMagnitude(magnitude: Double, fullScale: Double): Double {
		if (!magnitude.isFinite() || magnitude <= 0.0) return 0.0
		if (!fullScale.isFinite() || fullScale <= 0.0) return 0.0
		return (magnitude / fullScale).coerceIn(0.0, 1.0)
	}
}

/**
 * Platform-independent gyro/linear-acceleration scorer.
 *
 * Gyroscope activity is preferred. Linear acceleration contributes only while it is fresh and
 * available; if the gyroscope is absent, a fresh linear-acceleration signal can still provide a
 * weak activity hint. When only one sensor is fresh, that sensor is used alone as a fallback; the
 * configured two-sensor weight is not applied to a missing or stale companion. No axis is
 * interpreted as a camera axis and no signal is integrated.
 */
class PhoneMotionScoreLogic(
	private val config: PhoneMotionMonitorConfig = PhoneMotionMonitorConfig(),
) {
	private var gyroAtNanos: Long? = null
	private var linearAccelerationAtNanos: Long? = null
	private var gyroScoreEma: Double? = null
	private var linearAccelerationScoreEma: Double? = null

	/** Accepts one gyroscope vector in rad/s. Older same-sensor samples are ignored. */
	@Synchronized
	fun onGyroscopeSample(x: Float, y: Float, z: Float, timestampNanos: Long) {
		val magnitude = PhoneMotionMath.magnitude(x, y, z)
		val previousTimestamp = gyroAtNanos
		if (!magnitude.isFinite() ||
			(previousTimestamp != null && timestampNanos <= previousTimestamp)
		) {
			return
		}
		gyroAtNanos = timestampNanos
		val normalizedScore = PhoneMotionMath.normalizeMagnitude(
			magnitude,
			config.gyroFullScaleRadPerSecond,
		)
		gyroScoreEma = updateSensorEma(
			previousEma = gyroScoreEma,
			newScore = normalizedScore,
			previousTimestamp = previousTimestamp,
			newTimestamp = timestampNanos,
		)
	}

	/** Accepts one linear-acceleration vector in m/s^2. Older same-sensor samples are ignored. */
	@Synchronized
	fun onLinearAccelerationSample(x: Float, y: Float, z: Float, timestampNanos: Long) {
		val magnitude = PhoneMotionMath.magnitude(x, y, z)
		val previousTimestamp = linearAccelerationAtNanos
		if (!magnitude.isFinite() ||
			(previousTimestamp != null && timestampNanos <= previousTimestamp)
		) {
			return
		}
		linearAccelerationAtNanos = timestampNanos
		val normalizedScore = PhoneMotionMath.normalizeMagnitude(
			magnitude,
			config.linearAccelerationFullScaleMetersPerSecondSquared,
		)
		linearAccelerationScoreEma = updateSensorEma(
			previousEma = linearAccelerationScoreEma,
			newScore = normalizedScore,
			previousTimestamp = previousTimestamp,
			newTimestamp = timestampNanos,
		)
	}

	/** Returns a fresh normalized hint, or null when every received signal is stale/missing. */
	@Synchronized
	fun score(atNanos: Long): Double? {
		val gyroScore = gyroScoreEma.takeIf { isFresh(gyroAtNanos, atNanos) }
		val linearAccelerationScore =
			linearAccelerationScoreEma.takeIf { isFresh(linearAccelerationAtNanos, atNanos) }

		return combineFreshScores(gyroScore, linearAccelerationScore)
	}

	/** Clears sensor samples and smoothing history. */
	@Synchronized
	fun reset() {
		gyroAtNanos = null
		linearAccelerationAtNanos = null
		gyroScoreEma = null
		linearAccelerationScoreEma = null
	}

	private fun updateSensorEma(
		previousEma: Double?,
		newScore: Double,
		previousTimestamp: Long?,
		newTimestamp: Long,
	): Double {
		// A sensor's first sample after its timeout starts a new history. The other sensor keeps its
		// own EMA, so polling or a stale companion cannot create a hidden combined-score step.
		if (previousTimestamp != null && isBeyondTimeout(newTimestamp, previousTimestamp)) {
			return newScore
		}
		return previousEma?.let { previous ->
			previous + config.smoothingFactor * (newScore - previous)
		} ?: newScore
	}

	private fun combineFreshScores(gyroScore: Double?, linearAccelerationScore: Double?): Double? {
		return when {
			gyroScore != null && linearAccelerationScore != null ->
				(
					gyroScore * (1.0 - config.linearAccelerationWeight) +
						linearAccelerationScore * config.linearAccelerationWeight
				).coerceIn(0.0, 1.0)
			gyroScore != null -> gyroScore
			linearAccelerationScore != null -> linearAccelerationScore
			else -> null
		}
	}

	private fun isFresh(sampleAtNanos: Long?, nowNanos: Long): Boolean {
		if (sampleAtNanos == null || nowNanos < sampleAtNanos) return false
		return nowNanos <= safeAdd(sampleAtNanos, config.staleTimeoutNanos)
	}

	private fun isBeyondTimeout(nowNanos: Long, thenNanos: Long): Boolean =
		nowNanos > safeAdd(thenNanos, config.staleTimeoutNanos)

	private fun safeAdd(value: Long, increment: Long): Long =
		if (increment > 0L && value > Long.MAX_VALUE - increment) Long.MAX_VALUE else value + increment
}
