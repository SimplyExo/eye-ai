package com.algorithmic_alliance.eyeaiapp.inference.throttling

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveOdGateTest {
	@Test
	fun firstOpportunityIsImmediateAndConsuming() {
		val gate = gate(30.0)
		assertTrue(gate.tryAcquire(null, 0L).admitted)
		assertFalse(gate.tryAcquire(null, 1L).admitted)
	}

	@Test
	fun relativeQuietActiveAndBurstRatesFollowOneBudget() {
		val rates = ObjectDetectionPolicy.targetRates(30.0)
		assertEquals(10.5, rates.quietFps, 0.0)
		assertEquals(19.5, rates.activeFps, 0.0)
		assertEquals(30.0, rates.burstFps, 0.0)

		val gate = gate(30.0)
		assertEquals(interval(10.5), gate.mayAttempt(0L).inferenceIntervalNanos)
		gate.onVisualSample(0.5, 1L)
		assertEquals(InferenceMode.ACTIVE, gate.mode)
		assertEquals(interval(19.5), gate.mayAttempt(1L).inferenceIntervalNanos)
		gate.onVisualSample(0.9, 2L)
		assertEquals(InferenceMode.BURST, gate.mode)
		assertEquals(interval(30.0), gate.mayAttempt(2L).inferenceIntervalNanos)
	}

	@Test
	fun strongVisualChangeBypassesQuietIntervalButNotHardCap() {
		val gate = gate(30.0)
		assertTrue(gate.tryAcquire(null, 0L).admitted)
		gate.onVisualSample(0.9, 1L)
		assertFalse(gate.tryAcquire(null, interval(30.0) - 1L).admitted)
		assertTrue(gate.tryAcquire(null, interval(30.0)).admitted)
	}

	@Test
	fun motionEntersActiveButNeverBurst() {
		val gate = gate(30.0)
		assertTrue(gate.tryAcquire(0.7, 0L).admitted)
		assertEquals(InferenceMode.ACTIVE, gate.mode)
		gate.tryAcquire(1.0, interval(19.5))
		assertEquals(InferenceMode.ACTIVE, gate.mode)
	}

	@Test
	fun motionSampleCanRaiseCadenceBeforeModelAdmission() {
		val gate = gate(30.0)
		assertTrue(gate.tryAcquire(null, 0L).admitted)
		val afterSixtyMillis = 60_000_000L
		assertFalse(gate.mayAttempt(afterSixtyMillis).admitted)

		gate.onMotionSample(1.0, afterSixtyMillis)

		assertEquals(InferenceMode.ACTIVE, gate.mode)
		assertTrue(gate.mayAttempt(afterSixtyMillis).admitted)
	}

	@Test
	fun activeHysteresisAndLowActivityHoldPreventFlutter() {
		val gate = gate(30.0)
		gate.onVisualSample(0.5, 0L)
		assertEquals(InferenceMode.ACTIVE, gate.mode)
		gate.onVisualSample(0.2, 100_000_000L)
		assertEquals(InferenceMode.ACTIVE, gate.mode)
		gate.onVisualSample(0.05, 200_000_000L)
		assertEquals(InferenceMode.ACTIVE, gate.mode)
		gate.mayAttempt(1_199_999_999L)
		assertEquals(InferenceMode.ACTIVE, gate.mode)
		gate.mayAttempt(1_200_000_000L)
		assertEquals(InferenceMode.QUIET, gate.mode)
	}

	@Test
	fun burstHoldsThenReturnsThroughActive() {
		val gate = gate(30.0)
		gate.onVisualSample(0.9, 0L)
		gate.onVisualSample(0.5, 100_000_000L)
		gate.mayAttempt(499_999_999L)
		assertEquals(InferenceMode.BURST, gate.mode)
		gate.mayAttempt(500_000_000L)
		assertEquals(InferenceMode.ACTIVE, gate.mode)
	}

	@Test
	fun newStrongSampleExtendsAnExistingBurst() {
		val gate = gate(30.0)
		gate.onVisualSample(0.9, 0L)
		gate.onVisualSample(0.95, 400_000_000L)
		gate.mayAttempt(899_999_999L)
		assertEquals(InferenceMode.BURST, gate.mode)
		gate.mayAttempt(900_000_000L)
		assertEquals(InferenceMode.ACTIVE, gate.mode)
	}

	@Test
	fun longPauseYieldsOnlyOneSlot() {
		val gate = gate(30.0)
		assertTrue(gate.tryAcquire(null, 0L).admitted)
		assertTrue(gate.tryAcquire(null, 10_000_000_000L).admitted)
		assertFalse(gate.tryAcquire(null, 10_000_000_001L).admitted)
	}

	@Test
	fun budgetUpdatePreservesLastInferenceAndChangesEligibility() {
		val gate = gate(null)
		assertTrue(gate.tryAcquire(null, 100L).admitted)
		gate.updateBudget(6.0, 101L)
		val quietInterval = ObjectDetectionPolicy.budget(6.0).quietIntervalNanos
		assertFalse(gate.tryAcquire(null, 100L + quietInterval - 1L).admitted)
		assertTrue(gate.tryAcquire(null, 100L + quietInterval).admitted)
	}

	@Test
	fun unboundedModeAdmitsWithoutAnArtificialCadence() {
		val gate = gate(null)
		assertTrue(gate.tryAcquire(null, 0L).admitted)
		assertTrue(gate.tryAcquire(null, 0L).admitted)
		assertTrue(gate.tryAcquire(null, 1L).admitted)
	}

	@Test
	fun activityResetClearsModeButCanPreserveHardCapHistory() {
		val gate = gate(30.0)
		gate.onVisualSample(0.9, 0L)
		assertTrue(gate.tryAcquire(null, 0L).admitted)
		gate.resetActivity(1L)
		assertEquals(InferenceMode.QUIET, gate.mode)
		assertFalse(gate.tryAcquire(null, 2L).admitted)
		gate.resetActivity(3L, preserveLastInference = false)
		assertTrue(gate.tryAcquire(null, 3L).admitted)
	}

	@Test(expected = IllegalArgumentException::class)
	fun backwardsOperationTimeIsRejected() {
		val gate = gate(30.0, now = 10L)
		gate.mayAttempt(9L)
	}

	private fun gate(maxFps: Double?, now: Long = 0L) = AdaptiveOdGate(maxFps, now)

	private fun interval(fps: Double): Long = ObjectDetectionPolicy.intervalForFps(fps)
}
