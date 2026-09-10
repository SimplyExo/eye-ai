package com.algorithmic_alliance.eyeaiapp.inference.throttling

import java.util.Locale

data class InferenceTelemetrySnapshot(
    val visualChangeScore: Double?,
    val phoneMotionScore: Double?,
    val inferenceMode: InferenceMode,
    val modeChangeReason: String?,
    val lastInferenceIntervalNanos: Long?,
    val lastInferenceRuntimeNanos: Long?,
    val schedulerSkippedFrames: Long,
    val objectInferenceCount: Long,
    val recordedAtNanos: Long,
) {
    fun asLogLine(): String = String.format(
        Locale.US,
        "visual=%s phone=%s mode=%s reason=%s intervalMs=%s runtimeMs=%s skipped=%d inferences=%d",
        visualChangeScore?.let { "%.3f".format(Locale.US, it) } ?: "null",
        phoneMotionScore?.let { "%.3f".format(Locale.US, it) } ?: "null",
        inferenceMode,
        modeChangeReason ?: "none",
        lastInferenceIntervalNanos?.let { "%.1f".format(Locale.US, it / 1_000_000.0) } ?: "null",
        lastInferenceRuntimeNanos?.let { "%.1f".format(Locale.US, it / 1_000_000.0) } ?: "null",
        schedulerSkippedFrames,
        objectInferenceCount,
    )
}

class InferenceTelemetry(
    private val clock: MonotonicClock = AnalysisClock,
    private val logIntervalNanos: Long = 5_000_000_000L,
) {
    private val lock = Any()

    private var visualChangeScoreValue: Double? = null
    private var phoneMotionScoreValue: Double? = null
    private var inferenceModeValue = InferenceMode.QUIET
    private var modeChangeReasonValue: String? = null
    private var lastInferenceStartNanos: Long? = null
    private var lastInferenceIntervalNanosValue: Long? = null
    private var lastInferenceRuntimeNanosValue: Long? = null
    private var schedulerSkippedFramesValue = 0L
    private var objectInferenceCountValue = 0L
    private var activeInferenceStartNanos: Long? = null
    private var recordedAtNanosValue = 0L
    private var nextLogAtNanos = Long.MIN_VALUE

    init {
        require(logIntervalNanos > 0L) { "logIntervalNanos must be positive" }
    }

    fun startSession(atNanos: Long = clock.nowNanos()) = synchronized(lock) {
        visualChangeScoreValue = null
        phoneMotionScoreValue = null
        inferenceModeValue = InferenceMode.QUIET
        modeChangeReasonValue = "session_start"
        lastInferenceStartNanos = null
        lastInferenceIntervalNanosValue = null
        lastInferenceRuntimeNanosValue = null
        schedulerSkippedFramesValue = 0L
        objectInferenceCountValue = 0L
        activeInferenceStartNanos = null
        recordedAtNanosValue = atNanos
        nextLogAtNanos = Long.MIN_VALUE
    }

    fun resetActivity(reason: String, atNanos: Long = clock.nowNanos()) = synchronized(lock) {
        visualChangeScoreValue = null
        phoneMotionScoreValue = null
        inferenceModeValue = InferenceMode.QUIET
        modeChangeReasonValue = reason
        lastInferenceStartNanos = null
        lastInferenceIntervalNanosValue = null
        recordedAtNanosValue = atNanos
    }

    fun recordVisualChange(score: Double, atNanos: Long) = synchronized(lock) {
        visualChangeScoreValue = score
        recordedAtNanosValue = atNanos
    }

    fun recordDecision(
        decision: OdDecision,
        visualChangeScore: Double?,
        phoneMotionScore: Double?,
        atNanos: Long,
    ) = synchronized(lock) {
        visualChangeScoreValue = visualChangeScore
        phoneMotionScoreValue = phoneMotionScore
        val previousMode = inferenceModeValue
        inferenceModeValue = decision.mode
        if (previousMode != decision.mode) {
            modeChangeReasonValue = transitionReason(
                previousMode,
                decision.mode,
                visualChangeScore,
                phoneMotionScore,
            )
        }
        if (decision.admitted) {
            lastInferenceIntervalNanosValue = lastInferenceStartNanos?.let { atNanos - it }
                ?.takeIf { it >= 0L }
            lastInferenceStartNanos = atNanos
            activeInferenceStartNanos = atNanos
            objectInferenceCountValue++
        } else {
            schedulerSkippedFramesValue++
        }
        recordedAtNanosValue = atNanos
    }

    fun recordInferenceCompletion(startNanos: Long, completedNanos: Long) = synchronized(lock) {
        lastInferenceRuntimeNanosValue = (completedNanos - startNanos).takeIf { it >= 0L }
        if (activeInferenceStartNanos == startNanos) activeInferenceStartNanos = null
        recordedAtNanosValue = completedNanos
    }

    fun snapshot(atNanos: Long = clock.nowNanos()): InferenceTelemetrySnapshot = synchronized(lock) {
        snapshotLocked(atNanos)
    }

    fun pollLogSnapshot(atNanos: Long = clock.nowNanos()): InferenceTelemetrySnapshot? =
        synchronized(lock) {
            if (atNanos < nextLogAtNanos) return@synchronized null
            nextLogAtNanos = safeAdd(atNanos, logIntervalNanos)
            snapshotLocked(atNanos)
        }

    private fun snapshotLocked(atNanos: Long) = InferenceTelemetrySnapshot(
        visualChangeScoreValue,
        phoneMotionScoreValue,
        inferenceModeValue,
        modeChangeReasonValue,
        lastInferenceIntervalNanosValue,
        lastInferenceRuntimeNanosValue,
        schedulerSkippedFramesValue,
        objectInferenceCountValue,
        maxOf(atNanos, recordedAtNanosValue),
    )

    private fun transitionReason(
        previous: InferenceMode,
        next: InferenceMode,
        visualChangeScore: Double?,
        phoneMotionScore: Double?,
    ): String = when {
        next == InferenceMode.BURST -> "strong_visual_change"
        previous == InferenceMode.BURST && next == InferenceMode.ACTIVE -> "burst_hold_elapsed"
        next == InferenceMode.QUIET -> "low_activity_or_signal_timeout"
        phoneMotionScore != null && phoneMotionScore >= 0.5 -> "phone_motion_activity"
        visualChangeScore != null -> "visual_activity"
        else -> "activity_signal"
    }

    private fun safeAdd(value: Long, increment: Long): Long =
        if (value > Long.MAX_VALUE - increment) Long.MAX_VALUE else value + increment
}
