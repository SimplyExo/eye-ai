package com.algorithmic_alliance.eyeaiapp.inference.throttling

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class InferenceTelemetryTest {
	@Test
	fun oneDecisionCallRecordsAdmissionOrRejection() {
		var now = 0L
		val telemetry = InferenceTelemetry(MonotonicClock { now }, logIntervalNanos = 100L)
		telemetry.startSession()
		telemetry.recordVisualChange(0.7, 10L)
		telemetry.recordDecision(decision(InferenceMode.ACTIVE, true), 0.7, null, 20L)
		telemetry.recordInferenceCompletion(20L, 35L)
		telemetry.recordDecision(decision(InferenceMode.ACTIVE, false), 0.7, null, 50L)
		telemetry.recordDecision(decision(InferenceMode.ACTIVE, true), 0.7, null, 120L)
		telemetry.recordInferenceCompletion(120L, 150L)
		now = 150L

		val snapshot = telemetry.snapshot()
		assertEquals(0.7, snapshot.visualChangeScore ?: Double.NaN, 0.0)
		assertNull(snapshot.phoneMotionScore)
		assertEquals(InferenceMode.ACTIVE, snapshot.inferenceMode)
		assertEquals("visual_activity", snapshot.modeChangeReason)
		assertEquals(100L, snapshot.lastInferenceIntervalNanos)
		assertEquals(30L, snapshot.lastInferenceRuntimeNanos)
		assertEquals(1L, snapshot.schedulerSkippedFrames)
		assertEquals(2L, snapshot.objectInferenceCount)
		assertNotNull(telemetry.pollLogSnapshot())
		assertNull(telemetry.pollLogSnapshot(200L))
		assertNotNull(telemetry.pollLogSnapshot(250L))
	}

	@Test
	fun activityResetClearsSignalsButKeepsOperationCounters() {
		val telemetry = InferenceTelemetry(MonotonicClock { 0L })
		telemetry.startSession(0L)
		telemetry.recordDecision(decision(InferenceMode.QUIET, false), 0.9, null, 1L)
		telemetry.recordDecision(decision(InferenceMode.BURST, true), 0.9, null, 2L)
		telemetry.recordInferenceCompletion(2L, 3L)
		telemetry.resetActivity("stream_gap", 5L)

		val snapshot = telemetry.snapshot(5L)
		assertNull(snapshot.visualChangeScore)
		assertNull(snapshot.phoneMotionScore)
		assertEquals(InferenceMode.QUIET, snapshot.inferenceMode)
		assertEquals("stream_gap", snapshot.modeChangeReason)
		assertEquals(1L, snapshot.schedulerSkippedFrames)
		assertEquals(1L, snapshot.objectInferenceCount)
		assertNull(snapshot.lastInferenceIntervalNanos)
	}

	private fun decision(mode: InferenceMode, admitted: Boolean) = OdDecision(
		mode = mode,
		inferenceIntervalNanos = 1L,
		admitted = admitted,
		lastInferenceAtNanos = null,
		burstUntilNanos = null,
	)
}
