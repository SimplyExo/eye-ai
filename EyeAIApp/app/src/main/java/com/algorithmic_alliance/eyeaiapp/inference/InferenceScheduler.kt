package com.algorithmic_alliance.eyeaiapp.inference

import kotlin.math.ceil
import kotlin.time.Duration
import kotlin.time.Duration.Companion.nanoseconds

/** The qualitative cadence selected for object detection. */
enum class InferenceMode {
	QUIET,
	ACTIVE,
	BURST,
}

/** A monotone time source expressed in nanoseconds. */
fun interface MonotonicClock {
	fun nowNanos(): Long
}

/** Production clock for this platform-independent component. */
object SystemMonotonicClock : MonotonicClock {
	override fun nowNanos(): Long = System.nanoTime()
}

/**
 * All scheduler policy is supplied by the caller.
 *
 * The values are deliberately required instead of being presented as calibrated product
 * defaults. The fields named `Entry` and `Exit` provide hysteresis for ACTIVE; the two quiet
 * thresholds define when the low-activity hold can start.
 */
data class InferenceSchedulerConfig(
	val quietInterval: Duration,
	val activeInterval: Duration,
	val burstInterval: Duration,
	val quietHoldTime: Duration,
	val burstHoldTime: Duration,
	val signalTimeout: Duration,
	val activeVisualEntryThreshold: Double,
	val activeVisualExitThreshold: Double,
	val quietVisualThreshold: Double,
	val activeMotionEntryThreshold: Double,
	val activeMotionExitThreshold: Double,
	val quietMotionThreshold: Double,
	val burstVisualEntryThreshold: Double,
	val maxObjectDetectionRateHz: Double? = null,
) {
	init {
		requirePositiveFinite(quietInterval, "quietInterval")
		requirePositiveFinite(activeInterval, "activeInterval")
		requirePositiveFinite(burstInterval, "burstInterval")
		require(quietInterval >= activeInterval) {
			"quietInterval must be greater than or equal to activeInterval"
		}
		require(activeInterval >= burstInterval) {
			"activeInterval must be greater than or equal to burstInterval"
		}
		requireNonNegativeFinite(quietHoldTime, "quietHoldTime")
		requirePositiveFinite(burstHoldTime, "burstHoldTime")
		requirePositiveFinite(signalTimeout, "signalTimeout")

		requireScoreThreshold(activeVisualEntryThreshold, "activeVisualEntryThreshold")
		requireScoreThreshold(activeVisualExitThreshold, "activeVisualExitThreshold")
		requireScoreThreshold(quietVisualThreshold, "quietVisualThreshold")
		requireScoreThreshold(activeMotionEntryThreshold, "activeMotionEntryThreshold")
		requireScoreThreshold(activeMotionExitThreshold, "activeMotionExitThreshold")
		requireScoreThreshold(quietMotionThreshold, "quietMotionThreshold")
		requireScoreThreshold(burstVisualEntryThreshold, "burstVisualEntryThreshold")

		require(activeVisualEntryThreshold > activeVisualExitThreshold) {
			"activeVisualEntryThreshold must be greater than activeVisualExitThreshold"
		}
		require(activeMotionEntryThreshold > activeMotionExitThreshold) {
			"activeMotionEntryThreshold must be greater than activeMotionExitThreshold"
		}
		require(quietVisualThreshold <= activeVisualExitThreshold) {
			"quietVisualThreshold must not exceed activeVisualExitThreshold"
		}
		require(quietMotionThreshold <= activeMotionExitThreshold) {
			"quietMotionThreshold must not exceed activeMotionExitThreshold"
		}
		require(burstVisualEntryThreshold >= activeVisualEntryThreshold) {
			"burstVisualEntryThreshold must not be below activeVisualEntryThreshold"
		}

		maxObjectDetectionRateHz?.let {
			require(it.isFinite() && it > 0.0) {
				"maxObjectDetectionRateHz must be finite and greater than zero"
			}
		}
	}

	internal val quietIntervalNanos: Long
		get() = quietInterval.inWholeNanoseconds

	internal val activeIntervalNanos: Long
		get() = activeInterval.inWholeNanoseconds

	internal val burstIntervalNanos: Long
		get() = burstInterval.inWholeNanoseconds

	internal val quietHoldTimeNanos: Long
		get() = quietHoldTime.inWholeNanoseconds

	internal val burstHoldTimeNanos: Long
		get() = burstHoldTime.inWholeNanoseconds

	internal val signalTimeoutNanos: Long
		get() = signalTimeout.inWholeNanoseconds

	internal val maximumRateIntervalNanos: Long?
		get() = maxObjectDetectionRateHz?.let {
			ceil(NANOS_PER_SECOND.toDouble() / it).toLong().coerceAtLeast(1L)
		}

	private companion object {
		const val NANOS_PER_SECOND = 1_000_000_000L

		fun requirePositiveFinite(duration: Duration, name: String) {
			require(duration.isFinite() && duration > Duration.ZERO) {
				"$name must be finite and greater than zero"
			}
		}

		fun requireNonNegativeFinite(duration: Duration, name: String) {
			require(duration.isFinite() && duration >= Duration.ZERO) {
				"$name must be finite and non-negative"
			}
		}

		fun requireScoreThreshold(value: Double, name: String) {
			require(value.isFinite() && value in 0.0..1.0) {
				"$name must be finite and in the range 0.0..1.0"
			}
		}
	}
}

/** A snapshot of the scheduler at one monotone timestamp. */
data class InferenceSchedulerSnapshot(
	val mode: InferenceMode,
	val inferenceInterval: Duration,
	val canRunInference: Boolean,
	val lastInferenceAtNanos: Long?,
	val burstUntilNanos: Long?,
)

private data class TimedScore(
	val score: Double,
	val atNanos: Long,
)

/**
 * A small, platform-independent cadence controller for object detection.
 *
 * `tryAcquireInference` is the consuming operation. A successful call records the start time of
 * one inference. Callers can inspect the same decision with `canRunInference` or `snapshot`, but
 * those methods do not reserve additional slots. The class has no frame queue and never cancels
 * an inference that a caller has already started.
 */
class InferenceScheduler(
	config: InferenceSchedulerConfig,
	private val clock: MonotonicClock = SystemMonotonicClock,
) {
	private val stateLock = Any()
	private var config: InferenceSchedulerConfig = config
	private var lastOperationAtNanos: Long = clock.nowNanos()
	private var resetAtNanos: Long = lastOperationAtNanos

	private var visualSignal: TimedScore? = null
	private var lastAcceptedVisualSampleAtNanos: Long? = null
	private var phoneMotionSignal: TimedScore? = null
	private var lastAcceptedPhoneMotionSampleAtNanos: Long? = null
	private var phoneMotionAbsentSinceNanos: Long = resetAtNanos

	private var currentMode: InferenceMode = InferenceMode.QUIET
	private var lowActivitySinceNanos: Long? = resetAtNanos
	private var burstUntilNanos: Long? = null
	private var lastBurstSignalAtNanos: Long? = null
	private var immediateInferencePending: Boolean = false
	private var lastInferenceAtNanos: Long? = null

	/** The current mode evaluated against the clock. */
	val mode: InferenceMode
		get() = snapshot().mode

	/**
	 * Replaces both activity inputs with one timestamped sample.
	 *
	 * A null phone-motion score means that this sample contains no current phone-motion signal, so
	 * a previously stored motion score is cleared.
	 */
	fun updateSignals(
		visualChangeScore: Double,
		phoneMotionScore: Double? = null,
	): InferenceSchedulerSnapshot = synchronizedState {
		updateSignalsLocked(visualChangeScore, phoneMotionScore, clock.nowNanos())
	}

	/** Deterministic timestamped variant for tests and platform adapters with an owned time base. */
	fun updateSignals(
		visualChangeScore: Double,
		phoneMotionScore: Double? = null,
		atNanos: Long,
	): InferenceSchedulerSnapshot = synchronizedState {
		updateSignalsLocked(visualChangeScore, phoneMotionScore, atNanos)
	}

	/** Updates only the scene signal and preserves the latest phone-motion signal. */
	fun updateVisualChange(
		visualChangeScore: Double,
	): InferenceSchedulerSnapshot = synchronizedState {
		updateVisualChangeLocked(visualChangeScore, clock.nowNanos())
	}

	/** Deterministic timestamped variant for tests and platform adapters with an owned time base. */
	fun updateVisualChange(
		visualChangeScore: Double,
		atNanos: Long,
	): InferenceSchedulerSnapshot = synchronizedState {
		updateVisualChangeLocked(visualChangeScore, atNanos)
	}

	/** Updates only the phone-motion signal; null explicitly records that motion is absent. */
	fun updatePhoneMotion(
		phoneMotionScore: Double?,
	): InferenceSchedulerSnapshot = synchronizedState {
		updatePhoneMotionLocked(phoneMotionScore, clock.nowNanos())
	}

	/** Deterministic timestamped variant for tests and platform adapters with an owned time base. */
	fun updatePhoneMotion(
		phoneMotionScore: Double?,
		atNanos: Long,
	): InferenceSchedulerSnapshot = synchronizedState {
		updatePhoneMotionLocked(phoneMotionScore, atNanos)
	}

	/** Alias for callers that use activity terminology for a combined sample. */
	fun updateActivity(
		visualChangeScore: Double,
		phoneMotionScore: Double? = null,
	): InferenceSchedulerSnapshot = synchronizedState {
		updateSignalsLocked(visualChangeScore, phoneMotionScore, clock.nowNanos())
	}

	/** Deterministic timestamped variant for tests and platform adapters with an owned time base. */
	fun updateActivity(
		visualChangeScore: Double,
		phoneMotionScore: Double? = null,
		atNanos: Long,
	): InferenceSchedulerSnapshot = synchronizedState {
		updateSignalsLocked(visualChangeScore, phoneMotionScore, atNanos)
	}

	/** Replaces policy without clearing cadence or signal history. */
	fun updateConfig(newConfig: InferenceSchedulerConfig): InferenceSchedulerSnapshot =
		synchronizedState {
			updateConfigLocked(newConfig, clock.nowNanos())
		}

	/** Deterministic timestamped variant for tests and policy adapters. */
	fun updateConfig(
		newConfig: InferenceSchedulerConfig,
		atNanos: Long,
	): InferenceSchedulerSnapshot = synchronizedState {
		updateConfigLocked(newConfig, atNanos)
	}

	/**
	 * Returns whether a new inference would be allowed at [atNanos]. This is a non-consuming query.
	 * Use [tryAcquireInference] to consume the permission and record an inference start.
	 */
	fun canRunInference(): Boolean = synchronizedState {
		canRunInferenceLocked(clock.nowNanos())
	}

	/** Deterministic timestamped variant for tests and platform adapters with an owned time base. */
	fun canRunInference(atNanos: Long): Boolean = synchronizedState {
		canRunInferenceLocked(atNanos)
	}

	/**
	 * Atomically checks and consumes one inference opportunity.
	 *
	 * There is intentionally no catch-up counter: after a long frame pause, the successful call
	 * records the current time, so a second immediate call is not granted.
	 */
	fun tryAcquireInference(): Boolean = synchronizedState {
		tryAcquireInferenceLocked(clock.nowNanos())
	}

	/** Deterministic timestamped variant for tests and platform adapters with an owned time base. */
	fun tryAcquireInference(atNanos: Long): Boolean = synchronizedState {
		tryAcquireInferenceLocked(atNanos)
	}

	/** Returns the current decision and cadence without consuming an inference opportunity. */
	fun snapshot(): InferenceSchedulerSnapshot = synchronizedState {
		snapshotLocked(clock.nowNanos())
	}

	/** Deterministic timestamped variant for tests and platform adapters with an owned time base. */
	fun snapshot(atNanos: Long): InferenceSchedulerSnapshot = synchronizedState {
		snapshotLocked(atNanos)
	}

	/** Clears signals, mode hold, and cadence history. The next inference is immediate. */
	fun reset() {
		synchronizedState { resetLocked(clock.nowNanos()) }
	}

	/** Deterministic timestamped variant for tests and platform adapters with an owned time base. */
	fun reset(atNanos: Long) {
		synchronizedState { resetLocked(atNanos) }
	}

	/** Drops evidence/holds at a source or geometry boundary without bypassing the rate cap. */
	fun resetActivity() = synchronizedState {
		val lastStart = lastInferenceAtNanos
		resetLocked(clock.nowNanos())
		lastInferenceAtNanos = lastStart
	}

	private fun updateSignalsLocked(
		visualChangeScore: Double,
		phoneMotionScore: Double?,
		atNanos: Long,
	): InferenceSchedulerSnapshot {
		validateScore(visualChangeScore, "visualChangeScore")
		phoneMotionScore?.let { validateScore(it, "phoneMotionScore") }

		val now = observeSignalTime(atNanos)
		acceptVisualSample(visualChangeScore, atNanos)
		acceptPhoneMotionSample(phoneMotionScore, atNanos)

		refreshMode(now)
		return snapshotAt(now)
	}

	private fun updateVisualChangeLocked(
		visualChangeScore: Double,
		atNanos: Long,
	): InferenceSchedulerSnapshot {
		validateScore(visualChangeScore, "visualChangeScore")
		val now = observeSignalTime(atNanos)
		acceptVisualSample(visualChangeScore, atNanos)
		refreshMode(now)
		return snapshotAt(now)
	}

	private fun updatePhoneMotionLocked(
		phoneMotionScore: Double?,
		atNanos: Long,
	): InferenceSchedulerSnapshot {
		phoneMotionScore?.let { validateScore(it, "phoneMotionScore") }
		val now = observeSignalTime(atNanos)
		acceptPhoneMotionSample(phoneMotionScore, atNanos)
		refreshMode(now)
		return snapshotAt(now)
	}

	private fun updateConfigLocked(
		newConfig: InferenceSchedulerConfig,
		atNanos: Long,
	): InferenceSchedulerSnapshot {
		val now = observeOperationTime(atNanos)
		config = newConfig
		refreshMode(now)
		return snapshotAt(now)
	}

	private fun canRunInferenceLocked(atNanos: Long): Boolean {
		val now = observeOperationTime(atNanos)
		refreshMode(now)
		return isInferenceEligible(now)
	}

	private fun tryAcquireInferenceLocked(atNanos: Long): Boolean {
		val now = observeOperationTime(atNanos)
		refreshMode(now)
		if (!isInferenceEligible(now)) {
			return false
		}

		lastInferenceAtNanos = now
		immediateInferencePending = false
		return true
	}

	private fun snapshotLocked(atNanos: Long): InferenceSchedulerSnapshot {
		val now = observeOperationTime(atNanos)
		refreshMode(now)
		return snapshotAt(now)
	}

	private fun resetLocked(atNanos: Long) {
		val now = observeOperationTime(atNanos)
		resetAtNanos = now
		visualSignal = null
		lastAcceptedVisualSampleAtNanos = null
		phoneMotionSignal = null
		lastAcceptedPhoneMotionSampleAtNanos = null
		phoneMotionAbsentSinceNanos = now
		currentMode = InferenceMode.QUIET
		lowActivitySinceNanos = now
		burstUntilNanos = null
		lastBurstSignalAtNanos = null
		immediateInferencePending = false
		lastInferenceAtNanos = null
	}

	private fun acceptVisualSample(score: Double, atNanos: Long) {
		if (atNanos < resetAtNanos || atNanos < (lastAcceptedVisualSampleAtNanos ?: Long.MIN_VALUE)) {
			return
		}
		visualSignal = TimedScore(score, atNanos)
		lastAcceptedVisualSampleAtNanos = atNanos
	}

	private fun acceptPhoneMotionSample(score: Double?, atNanos: Long) {
		val lowerBound = maxOf(
			resetAtNanos,
			lastAcceptedPhoneMotionSampleAtNanos ?: Long.MIN_VALUE,
			phoneMotionAbsentSinceNanos,
		)
		if (atNanos < lowerBound) return

		if (score == null) {
			phoneMotionSignal = null
			phoneMotionAbsentSinceNanos = atNanos
		} else {
			phoneMotionSignal = TimedScore(score, atNanos)
			lastAcceptedPhoneMotionSampleAtNanos = atNanos
		}
	}

	private fun snapshotAt(now: Long): InferenceSchedulerSnapshot =
		InferenceSchedulerSnapshot(
			mode = currentMode,
			inferenceInterval = effectiveInterval(currentMode).nanoseconds,
			canRunInference = isInferenceEligible(now),
			lastInferenceAtNanos = lastInferenceAtNanos,
			burstUntilNanos = burstUntilNanos,
		)

	private fun refreshMode(now: Long) {
		val visual = freshScore(visualSignal, now)
		val motion = freshScore(phoneMotionSignal, now)
		updateLowActivity(now, visual, motion)

		val strongVisualEvent = strongVisualEvent(now)
		if (strongVisualEvent != null) {
			if (currentMode == InferenceMode.BURST) {
				extendBurst(now, strongVisualEvent.atNanos)
			} else {
				enterBurst(now, strongVisualEvent.atNanos)
			}
			return
		}

		when (currentMode) {
			InferenceMode.QUIET -> {
				if (hasActiveEntryEvidence(visual, motion)) {
					enterActive()
				}
			}

			InferenceMode.ACTIVE -> {
				if (isQuietReady(now)) {
					enterQuiet()
				}
			}

			InferenceMode.BURST -> {
				val holdUntil = burstUntilNanos ?: now
				if (now < holdUntil) {
					return
				}

				burstUntilNanos = null
				if (hasActiveExitEvidence(visual, motion)) {
					enterActive()
				} else if (isQuietReady(now)) {
					enterQuiet()
				} else {
					// The signal is in the hysteresis band. Stay at the safer active cadence until
					// low activity has been observed for the configured hold time.
					enterActive()
				}
			}
		}
	}

	private fun updateLowActivity(now: Long, visual: Double?, motion: Double?) {
		if (!isLowActivity(visual, motion)) {
			lowActivitySinceNanos = null
			return
		}

		val estimatedStart = estimateLowActivityStart(now)
		lowActivitySinceNanos = minOf(lowActivitySinceNanos ?: estimatedStart, estimatedStart)
	}

	private fun estimateLowActivityStart(now: Long): Long {
		val visualLowSince = lowSinceFor(visualSignal, config.quietVisualThreshold, resetAtNanos)
		val motionLowSince = lowSinceFor(
			phoneMotionSignal,
			config.quietMotionThreshold,
			phoneMotionAbsentSinceNanos,
		)
		return maxOf(resetAtNanos, minOf(now, maxOf(visualLowSince, motionLowSince)))
	}

	private fun lowSinceFor(signal: TimedScore?, quietThreshold: Double, absentSince: Long): Long {
		if (signal == null) {
			return absentSince
		}
		return if (signal.score <= quietThreshold) {
			signal.atNanos
		} else {
			safeAdd(signal.atNanos, config.signalTimeoutNanos)
		}
	}

	private fun strongVisualEvent(now: Long): TimedScore? {
		val visual = visualSignal ?: return null
		if (freshScore(visual, now) == null || visual.score < config.burstVisualEntryThreshold) {
			return null
		}
		return if (visual.atNanos > (lastBurstSignalAtNanos ?: Long.MIN_VALUE)) visual else null
	}

	private fun hasActiveEntryEvidence(visual: Double?, motion: Double?): Boolean =
		(visual != null && visual >= config.activeVisualEntryThreshold) ||
			(motion != null && motion >= config.activeMotionEntryThreshold)

	private fun hasActiveExitEvidence(visual: Double?, motion: Double?): Boolean =
		(visual != null && visual >= config.activeVisualExitThreshold) ||
			(motion != null && motion >= config.activeMotionExitThreshold)

	private fun isLowActivity(visual: Double?, motion: Double?): Boolean =
		(visual == null || visual <= config.quietVisualThreshold) &&
			(motion == null || motion <= config.quietMotionThreshold)

	private fun isQuietReady(now: Long): Boolean {
		val lowSince = lowActivitySinceNanos ?: return false
		return elapsedNanos(now, lowSince) >= config.quietHoldTimeNanos
	}

	private fun freshScore(signal: TimedScore?, now: Long): Double? {
		if (signal == null || signal.atNanos > now) {
			return null
		}
		return if (elapsedNanos(now, signal.atNanos) <= config.signalTimeoutNanos) {
			signal.score
		} else {
			null
		}
	}

	private fun enterQuiet() {
		currentMode = InferenceMode.QUIET
		burstUntilNanos = null
	}

	private fun enterActive() {
		currentMode = InferenceMode.ACTIVE
		burstUntilNanos = null
	}

	private fun enterBurst(now: Long, signalAtNanos: Long) {
		currentMode = InferenceMode.BURST
		burstUntilNanos = safeAdd(now, config.burstHoldTimeNanos)
		lastBurstSignalAtNanos = signalAtNanos
		immediateInferencePending = true
	}

	private fun extendBurst(now: Long, signalAtNanos: Long) {
		currentMode = InferenceMode.BURST
		burstUntilNanos = maxOf(burstUntilNanos ?: now, safeAdd(now, config.burstHoldTimeNanos))
		lastBurstSignalAtNanos = signalAtNanos
	}

	private fun isInferenceEligible(now: Long): Boolean {
		val lastInference = lastInferenceAtNanos ?: return true
		// BURST is an urgency override for the policy interval, never for the global rate cap.
		val maximumRateInterval = config.maximumRateIntervalNanos
		if (maximumRateInterval != null &&
			elapsedNanos(now, lastInference) < maximumRateInterval
		) {
			return false
		}
		if (immediateInferencePending) {
			return true
		}
		return elapsedNanos(now, lastInference) >= effectiveInterval(currentMode)
	}

	private fun effectiveInterval(mode: InferenceMode): Long {
		val modeInterval = when (mode) {
			InferenceMode.QUIET -> config.quietIntervalNanos
			InferenceMode.ACTIVE -> config.activeIntervalNanos
			InferenceMode.BURST -> config.burstIntervalNanos
		}
		return maxOf(modeInterval, config.maximumRateIntervalNanos ?: 0L)
	}

	private fun observeSignalTime(atNanos: Long): Long {
		if (atNanos > lastOperationAtNanos) {
			lastOperationAtNanos = atNanos
		}
		return lastOperationAtNanos
	}

	private fun observeOperationTime(atNanos: Long): Long {
		require(atNanos >= lastOperationAtNanos) {
			"monotonic time moved backwards: $atNanos < $lastOperationAtNanos"
		}
		lastOperationAtNanos = atNanos
		return atNanos
	}

	private fun validateScore(score: Double, name: String) {
		require(score.isFinite() && score in 0.0..1.0) {
			"$name must be finite and in the range 0.0..1.0"
		}
	}

	private fun elapsedNanos(now: Long, then: Long): Long {
		require(now >= then) { "monotonic time moved backwards: $now < $then" }
		return now - then
	}

	private fun safeAdd(value: Long, increment: Long): Long =
		if (increment > 0L && value > Long.MAX_VALUE - increment) Long.MAX_VALUE else value + increment

	private inline fun <T> synchronizedState(block: () -> T): T = synchronized(stateLock, block)
}
