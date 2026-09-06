package com.algorithmic_alliance.eyeaiapp.inference

import android.os.SystemClock
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.nanoseconds
import kotlin.math.ceil

/** One local Android time domain. Capture/RTP timestamps are metadata only. */
object AnalysisClock : MonotonicClock, PhoneMotionClock {
    override fun nowNanos(): Long = SystemClock.elapsedRealtimeNanos()
}

/**
 * V1 adaptive object-detection budget policy.
 *
 * The setting is a hard maximum budget. Mode rates are derived from that budget,
 * so changing the budget changes all three mode intervals together.
 */
object ObjectDetectionV1Policy {
    const val STREAM_GAP_NANOS = 5_000_000_000L
    const val RESULT_TTL_NANOS = 1_000_000_000L
    const val DEPTH_OD_MAX_SKEW_NANOS = 250_000_000L

    const val MIN_USER_MAX_FPS = 5.0
    /** Upper bound of the enabled Object-Detection slider and persisted setting. */
    const val MAX_USER_MAX_FPS = 60.0
    const val QUIET_RATE_RATIO = 0.35
    const val ACTIVE_RATE_RATIO = 0.65

    data class TargetRates(
        val budgetFps: Double,
        val quietFps: Double,
        val activeFps: Double,
        val burstFps: Double,
    )

    /**
     * Returns the effective enabled-limiter budget. Values below the user floor
     * are safely raised to 5 and values above the slider ceiling are capped at 60.
     */
    fun targetRates(userMaxFps: Double): TargetRates {
        require(userMaxFps.isFinite() && userMaxFps > 0.0) {
            "userMaxFps must be finite and greater than zero"
        }
        val budget = userMaxFps.coerceIn(MIN_USER_MAX_FPS, MAX_USER_MAX_FPS)
        return TargetRates(
            budgetFps = budget,
            quietFps = minOf(budget, maxOf(MIN_USER_MAX_FPS, QUIET_RATE_RATIO * budget)),
            activeFps = minOf(budget, maxOf(MIN_USER_MAX_FPS, ACTIVE_RATE_RATIO * budget)),
            burstFps = budget,
        )
    }

    private fun intervalForFps(fps: Double) =
        ceil(1_000_000_000.0 / fps).toLong().coerceAtLeast(1L).nanoseconds

    /** Creates the scheduler configuration from one effective budget. */
    fun config(maxRateHz: Double? = null): InferenceSchedulerConfig {
        // With the user limiter disabled, keep activity classification and latest-frame
        // semantics but introduce no artificial cadence delay in any mode.
        val rates = maxRateHz?.let(::targetRates)
        val unlimitedInterval = 1.nanoseconds
        return InferenceSchedulerConfig(
            // Ceil intervals keep every delivered rate at or below its target.
            quietInterval = rates?.let { intervalForFps(it.quietFps) } ?: unlimitedInterval,
            activeInterval = rates?.let { intervalForFps(it.activeFps) } ?: unlimitedInterval,
            burstInterval = rates?.let { intervalForFps(it.burstFps) } ?: unlimitedInterval,
            quietHoldTime = 1_000.milliseconds,
            burstHoldTime = 500.milliseconds,
            signalTimeout = 500.milliseconds,
            activeVisualEntryThreshold = 0.4,
            activeVisualExitThreshold = 0.25,
            quietVisualThreshold = 0.1,
            activeMotionEntryThreshold = 0.5,
            activeMotionExitThreshold = 0.3,
            quietMotionThreshold = 0.1,
            burstVisualEntryThreshold = 0.8,
            // This is the same budget represented by the mode intervals. Keeping
            // the scheduler's hard cap equal to BURST makes the budget invariant
            // explicit without adding another post-inference limiter.
            maxObjectDetectionRateHz = rates?.budgetFps,
        )
    }
}
