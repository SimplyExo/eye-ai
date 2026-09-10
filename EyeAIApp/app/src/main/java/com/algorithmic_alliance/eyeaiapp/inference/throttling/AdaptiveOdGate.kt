package com.algorithmic_alliance.eyeaiapp.inference.throttling

enum class InferenceMode {
	QUIET,
	ACTIVE,
	BURST,
}

data class OdDecision(
	val mode: InferenceMode,
	val inferenceIntervalNanos: Long,
	val admitted: Boolean,
	val lastInferenceAtNanos: Long?,
	val burstUntilNanos: Long?,
)

private data class TimedScore(val score: Double, val atNanos: Long)

class AdaptiveOdGate(
	maxRateHz: Double?,
	nowNanos: Long,
) {
	private var budget = ObjectDetectionPolicy.budget(maxRateHz)
	private var lastOperationAtNanos = nowNanos
	private var resetAtNanos = nowNanos
	private var visualSignal: TimedScore? = null
	private var motionSignal: TimedScore? = null
	private var motionAbsentSinceNanos = nowNanos
	private var currentMode = InferenceMode.QUIET
	private var lowActivitySinceNanos: Long? = nowNanos
	private var burstUntilNanos: Long? = null
	private var lastBurstSignalAtNanos: Long? = null
	private var immediateInferencePending = false
	private var lastInferenceAtNanos: Long? = null

	val mode: InferenceMode
		get() = currentMode

	fun onVisualSample(score: Double, nowNanos: Long) {
		validateScore(score, "visualChangeScore")
		val now = observe(nowNanos)
		visualSignal = TimedScore(score, now)
		refreshMode(now)
	}

	fun onMotionSample(score: Double?, nowNanos: Long) {
		score?.let { validateScore(it, "phoneMotionScore") }
		val now = observe(nowNanos)
		updateMotion(score, now)
		refreshMode(now)
	}

	fun mayAttempt(nowNanos: Long): OdDecision {
		val now = observe(nowNanos)
		refreshMode(now)
		return decision(isEligible(now))
	}

	fun tryAcquire(phoneMotionScore: Double?, nowNanos: Long): OdDecision {
		onMotionSample(phoneMotionScore, nowNanos)
		val now = nowNanos
		val admitted = isEligible(now)
		if (admitted) {
			lastInferenceAtNanos = now
			immediateInferencePending = false
		}
		return decision(admitted)
	}

	fun updateBudget(maxRateHz: Double?, nowNanos: Long) {
		val now = observe(nowNanos)
		budget = ObjectDetectionPolicy.budget(maxRateHz)
		refreshMode(now)
	}

	fun resetActivity(
		nowNanos: Long,
		preserveLastInference: Boolean = true,
	) {
		val now = observe(nowNanos)
		val previousInference = lastInferenceAtNanos.takeIf { preserveLastInference }
		resetAtNanos = now
		visualSignal = null
		motionSignal = null
		motionAbsentSinceNanos = now
		currentMode = InferenceMode.QUIET
		lowActivitySinceNanos = now
		burstUntilNanos = null
		lastBurstSignalAtNanos = null
		immediateInferencePending = false
		lastInferenceAtNanos = previousInference
	}

	private fun updateMotion(score: Double?, now: Long) {
		if (score == null) {
			if (motionSignal != null) motionAbsentSinceNanos = now
			motionSignal = null
		} else {
			motionSignal = TimedScore(score, now)
		}
	}

	private fun refreshMode(now: Long) {
		val visual = freshScore(visualSignal, now)
		val motion = freshScore(motionSignal, now)
		updateLowActivity(now, visual, motion)

		val strongVisual = visualSignal?.takeIf {
			visual != null && it.score >= ObjectDetectionPolicy.BURST_VISUAL_ENTRY &&
				it.atNanos > (lastBurstSignalAtNanos ?: Long.MIN_VALUE)
		}
		if (strongVisual != null) {
			if (currentMode == InferenceMode.BURST) {
				burstUntilNanos = maxOf(
					burstUntilNanos ?: now,
					safeAdd(now, ObjectDetectionPolicy.BURST_HOLD_NANOS),
				)
			} else {
				currentMode = InferenceMode.BURST
				burstUntilNanos = safeAdd(now, ObjectDetectionPolicy.BURST_HOLD_NANOS)
				immediateInferencePending = true
			}
			lastBurstSignalAtNanos = strongVisual.atNanos
			return
		}

		when (currentMode) {
			InferenceMode.QUIET -> if (hasActiveEntry(visual, motion)) enterActive()
			InferenceMode.ACTIVE -> if (isQuietReady(now)) enterQuiet()
			InferenceMode.BURST -> {
				if (now < (burstUntilNanos ?: now)) return
				if (hasActiveExit(visual, motion) || !isQuietReady(now)) enterActive()
				else enterQuiet()
			}
		}
	}

	private fun updateLowActivity(now: Long, visual: Double?, motion: Double?) {
		if (!isLowActivity(visual, motion)) {
			lowActivitySinceNanos = null
			return
		}
		val visualLowSince = lowSinceFor(
			visualSignal,
			ObjectDetectionPolicy.QUIET_VISUAL,
			resetAtNanos,
		)
		val motionLowSince = lowSinceFor(
			motionSignal,
			ObjectDetectionPolicy.QUIET_MOTION,
			motionAbsentSinceNanos,
		)
		val estimatedStart = maxOf(resetAtNanos, minOf(now, maxOf(visualLowSince, motionLowSince)))
		lowActivitySinceNanos = minOf(lowActivitySinceNanos ?: estimatedStart, estimatedStart)
	}

	private fun lowSinceFor(signal: TimedScore?, quietThreshold: Double, absentSince: Long): Long =
		when {
			signal == null -> absentSince
			signal.score <= quietThreshold -> signal.atNanos
			else -> safeAdd(signal.atNanos, ObjectDetectionPolicy.SIGNAL_TIMEOUT_NANOS)
		}

	private fun freshScore(signal: TimedScore?, now: Long): Double? = signal?.score?.takeIf {
		now - signal.atNanos <= ObjectDetectionPolicy.SIGNAL_TIMEOUT_NANOS
	}

	private fun hasActiveEntry(visual: Double?, motion: Double?): Boolean =
		(visual != null && visual >= ObjectDetectionPolicy.ACTIVE_VISUAL_ENTRY) ||
			(motion != null && motion >= ObjectDetectionPolicy.ACTIVE_MOTION_ENTRY)

	private fun hasActiveExit(visual: Double?, motion: Double?): Boolean =
		(visual != null && visual >= ObjectDetectionPolicy.ACTIVE_VISUAL_EXIT) ||
			(motion != null && motion >= ObjectDetectionPolicy.ACTIVE_MOTION_EXIT)

	private fun isLowActivity(visual: Double?, motion: Double?): Boolean =
		(visual == null || visual <= ObjectDetectionPolicy.QUIET_VISUAL) &&
			(motion == null || motion <= ObjectDetectionPolicy.QUIET_MOTION)

	private fun isQuietReady(now: Long): Boolean = lowActivitySinceNanos?.let {
		now - it >= ObjectDetectionPolicy.QUIET_HOLD_NANOS
	} ?: false

	private fun enterActive() {
		currentMode = InferenceMode.ACTIVE
		burstUntilNanos = null
	}

	private fun enterQuiet() {
		currentMode = InferenceMode.QUIET
		burstUntilNanos = null
	}

	private fun isEligible(now: Long): Boolean {
		val previous = lastInferenceAtNanos ?: return true
		val elapsed = now - previous
		if (elapsed < (budget.maximumRateIntervalNanos ?: 0L)) return false
		return immediateInferencePending || elapsed >= modeInterval()
	}

	private fun modeInterval(): Long = when (currentMode) {
		InferenceMode.QUIET -> budget.quietIntervalNanos
		InferenceMode.ACTIVE -> budget.activeIntervalNanos
		InferenceMode.BURST -> budget.burstIntervalNanos
	}

	private fun decision(admitted: Boolean) = OdDecision(
		mode = currentMode,
		inferenceIntervalNanos = modeInterval(),
		admitted = admitted,
		lastInferenceAtNanos = lastInferenceAtNanos,
		burstUntilNanos = burstUntilNanos,
	)

	private fun observe(nowNanos: Long): Long {
		require(nowNanos >= lastOperationAtNanos) {
			"monotonic time moved backwards: $nowNanos < $lastOperationAtNanos"
		}
		lastOperationAtNanos = nowNanos
		return nowNanos
	}

	private fun validateScore(score: Double, name: String) {
		require(score.isFinite() && score in 0.0..1.0) {
			"$name must be finite and in the range 0.0..1.0"
		}
	}

	private fun safeAdd(value: Long, increment: Long): Long =
		if (value > Long.MAX_VALUE - increment) Long.MAX_VALUE else value + increment
}
