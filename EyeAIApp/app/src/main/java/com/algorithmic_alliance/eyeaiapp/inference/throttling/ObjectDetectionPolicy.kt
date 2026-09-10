package com.algorithmic_alliance.eyeaiapp.inference.throttling

import com.algorithmic_alliance.eyeaiapp.Settings
import kotlin.math.ceil

class ObjectDetectionBudget internal constructor(
	val maxFps: Double?,
	val quietIntervalNanos: Long,
	val activeIntervalNanos: Long,
	val burstIntervalNanos: Long,
) {
	val maximumRateIntervalNanos = maxFps?.let(ObjectDetectionPolicy::intervalForFps)
}

object ObjectDetectionPolicy {
	const val STREAM_GAP_NANOS = 5_000_000_000L
	const val RESULT_TTL_NANOS = 1_000_000_000L
	const val DEPTH_OD_MAX_SKEW_NANOS = 250_000_000L

	const val QUIET_RATE_RATIO = 0.35
	const val ACTIVE_RATE_RATIO = 0.65

	internal const val QUIET_HOLD_NANOS = 1_000_000_000L
	internal const val BURST_HOLD_NANOS = 500_000_000L
	internal const val SIGNAL_TIMEOUT_NANOS = 500_000_000L
	internal const val ACTIVE_VISUAL_ENTRY = 0.4
	internal const val ACTIVE_VISUAL_EXIT = 0.25
	internal const val QUIET_VISUAL = 0.1
	internal const val ACTIVE_MOTION_ENTRY = 0.5
	internal const val ACTIVE_MOTION_EXIT = 0.3
	internal const val QUIET_MOTION = 0.1
	internal const val BURST_VISUAL_ENTRY = 0.8

	data class TargetRates(
		val budgetFps: Double,
		val quietFps: Double,
		val activeFps: Double,
		val burstFps: Double,
	)

	fun targetRates(userMaxFps: Double): TargetRates {
		require(userMaxFps.isFinite() && userMaxFps > 0.0) {
			"userMaxFps must be finite and greater than zero"
		}
		val minimumFps = Settings.MIN_OBJECT_DETECTION_FRAME_RATE.toDouble()
		val maximumFps = Settings.MAX_OBJECT_DETECTION_FRAME_RATE.toDouble()
		val budget = userMaxFps.coerceIn(minimumFps, maximumFps)
		return TargetRates(
			budgetFps = budget,
			quietFps = minOf(budget, maxOf(minimumFps, QUIET_RATE_RATIO * budget)),
			activeFps = minOf(budget, maxOf(minimumFps, ACTIVE_RATE_RATIO * budget)),
			burstFps = budget,
		)
	}

	fun budget(maxRateHz: Double?): ObjectDetectionBudget {
		val rates = maxRateHz?.let(::targetRates)
		return ObjectDetectionBudget(
			maxFps = rates?.budgetFps,
			quietIntervalNanos = rates?.quietFps?.let(::intervalForFps) ?: 0L,
			activeIntervalNanos = rates?.activeFps?.let(::intervalForFps) ?: 0L,
			burstIntervalNanos = rates?.burstFps?.let(::intervalForFps) ?: 0L,
		)
	}

	internal fun intervalForFps(fps: Double): Long =
		ceil(1_000_000_000.0 / fps).toLong().coerceAtLeast(1L)
}
