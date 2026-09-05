package com.algorithmic_alliance.eyeaiapp.inference

import android.os.SystemClock
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.nanoseconds

/** One local Android time domain. Capture/RTP timestamps are metadata only. */
object AnalysisClock : MonotonicClock, PhoneMotionClock {
    override fun nowNanos(): Long = SystemClock.elapsedRealtimeNanos()
}

/** Technical starting values, not product calibration or guaranteed delivered inference rates. */
object ObjectDetectionV1Policy {
    const val STREAM_GAP_NANOS = 5_000_000_000L
    const val RESULT_TTL_NANOS = 1_000_000_000L
    const val DEPTH_OD_MAX_SKEW_NANOS = 250_000_000L

    // Ceil intervals keep rates at or below 3/7/15 Hz. A lower user cap always wins;
    // below 3 Hz (or under slow inference) normal tentative-track confirmation is not guaranteed.
    fun config(maxRateHz: Double? = null) = InferenceSchedulerConfig(
        quietInterval = 333_333_334.nanoseconds,
        activeInterval = 142_857_143.nanoseconds,
        burstInterval = 66_666_667.nanoseconds,
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
        maxObjectDetectionRateHz = maxRateHz,
    )
}
