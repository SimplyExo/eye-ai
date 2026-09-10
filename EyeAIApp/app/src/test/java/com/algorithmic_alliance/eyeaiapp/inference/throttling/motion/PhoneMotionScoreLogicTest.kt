package com.algorithmic_alliance.eyeaiapp.inference.throttling.motion

import kotlin.time.Duration.Companion.milliseconds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class PhoneMotionScoreLogicTest {
	@Test
	fun magnitudeUsesAllThreeAxes() {
		assertEquals(5.0, PhoneMotionMath.magnitude(3f, 4f, 0f), 0.000001)
	}

	@Test
	fun normalizationClampsToZeroAndOne() {
		assertEquals(0.0, PhoneMotionMath.normalizeMagnitude(-1.0, 2.0), 0.0)
		assertEquals(0.5, PhoneMotionMath.normalizeMagnitude(1.0, 2.0), 0.0)
		assertEquals(1.0, PhoneMotionMath.normalizeMagnitude(10.0, 2.0), 0.0)
	}

	@Test
	fun smoothingFollowsAChangedGyroMagnitudeGradually() {
		val logic = logic(smoothingFactor = 0.5, gyroFullScale = 1.0)

		logic.onGyroscopeSample(0f, 0f, 0f, timestampNanos = 0L)
		assertEquals(0.0, logic.score(0L)!!, 0.0)
		logic.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 1L)
		assertEquals(0.5, logic.score(1L)!!, 0.000001)
		logic.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 2L)
		assertEquals(0.75, logic.score(2L)!!, 0.000001)
	}

	@Test
	fun linearAccelerationIsASupplementalOrFallbackSignal() {
		val logic = logic(smoothingFactor = 1.0, gyroFullScale = 1.0, accelerationFullScale = 3.0)

		logic.onLinearAccelerationSample(0f, 3f, 0f, timestampNanos = 0L)

		assertEquals(1.0, logic.score(0L)!!, 0.0)
	}

	@Test
	fun combinedScoreStaysNormalizedWhenBothSensorsSaturate() {
		val logic = logic(smoothingFactor = 1.0, gyroFullScale = 1.0, accelerationFullScale = 1.0)

		logic.onGyroscopeSample(10f, 0f, 0f, timestampNanos = 0L)
		logic.onLinearAccelerationSample(0f, 10f, 0f, timestampNanos = 0L)

		val score = logic.score(0L)
		assertTrue(score != null && score in 0.0..1.0)
	}

	@Test
	fun staleSensorDataReturnsNull() {
		val logic = logic(staleTimeout = 100.milliseconds)

		logic.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 0L)

		assertEquals(0.5, logic.score(100.milliseconds.inWholeNanoseconds)!!, 0.0)
		assertNull(logic.score(101.milliseconds.inWholeNanoseconds))
	}

	@Test
	fun pollingFrequencyDoesNotChangeTheScoreAfterASensorTimeout() {
		fun runSequence(pollAt600Milliseconds: Boolean): Double? {
			val logic = logic(staleTimeout = 500.milliseconds, smoothingFactor = 0.25, gyroFullScale = 1.0)
			logic.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 0L)
			if (pollAt600Milliseconds) {
				assertNull(logic.score(600.milliseconds.inWholeNanoseconds))
			}
			logic.onGyroscopeSample(0f, 0f, 0f, timestampNanos = 1_000.milliseconds.inWholeNanoseconds)
			return logic.score(1_000.milliseconds.inWholeNanoseconds)
		}

		val withoutPoll = runSequence(pollAt600Milliseconds = false)!!
		val withPoll = runSequence(pollAt600Milliseconds = true)!!
		assertEquals(withoutPoll, withPoll, 0.0)
		assertEquals(0.0, runSequence(pollAt600Milliseconds = false)!!, 0.0)
	}

	@Test
	fun scorePollingOnlyChecksFreshnessAndDoesNotPerformAnotherEmaStep() {
		val logic = logic(staleTimeout = 500.milliseconds, smoothingFactor = 0.5, gyroFullScale = 1.0)
		logic.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 0L)
		logic.onGyroscopeSample(
			0f,
			0f,
			0f,
			timestampNanos = 100.milliseconds.inWholeNanoseconds,
		)

		assertEquals(0.5, logic.score(100.milliseconds.inWholeNanoseconds)!!, 0.0)
		assertEquals(0.5, logic.score(100.milliseconds.inWholeNanoseconds)!!, 0.0)
		assertNull(logic.score(601.milliseconds.inWholeNanoseconds))
		assertNull(logic.score(601.milliseconds.inWholeNanoseconds))
	}

	@Test
	fun firstSampleAfterTimeoutStartsAFreshSensorEma() {
		val logic = logic(staleTimeout = 500.milliseconds, smoothingFactor = 0.25, gyroFullScale = 1.0)
		logic.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 0L)
		logic.onGyroscopeSample(
			0f,
			0f,
			0f,
			timestampNanos = 1_000.milliseconds.inWholeNanoseconds,
		)

		assertEquals(0.0, logic.score(1_000.milliseconds.inWholeNanoseconds)!!, 0.0)
	}

	@Test
	fun eachSensorIsIndependentlyFreshnessFiltered() {
		val staleAt = 600.milliseconds.inWholeNanoseconds
		val gyroStale = logic(staleTimeout = 500.milliseconds, smoothingFactor = 1.0, gyroFullScale = 1.0, accelerationFullScale = 1.0)
		gyroStale.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 0L)
		gyroStale.onLinearAccelerationSample(1f, 0f, 0f, timestampNanos = staleAt)
		assertEquals(1.0, gyroStale.score(staleAt)!!, 0.0)

		val accelerationStale = logic(staleTimeout = 500.milliseconds, smoothingFactor = 1.0, gyroFullScale = 1.0, accelerationFullScale = 1.0)
		accelerationStale.onLinearAccelerationSample(1f, 0f, 0f, timestampNanos = 0L)
		accelerationStale.onGyroscopeSample(1f, 0f, 0f, timestampNanos = staleAt)
		assertEquals(1.0, accelerationStale.score(staleAt)!!, 0.0)
	}

	@Test
	fun bothStaleSensorsProduceNull() {
		val logic = logic(staleTimeout = 500.milliseconds, smoothingFactor = 1.0)
		logic.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 0L)
		logic.onLinearAccelerationSample(1f, 0f, 0f, timestampNanos = 0L)

		assertNull(logic.score(501.milliseconds.inWholeNanoseconds))
	}

	@Test
	fun zeroAlphaIsRejectedAndOneAlphaIsValid() {
		try {
			PhoneMotionMonitorConfig(smoothingFactor = 0.0)
			fail("expected zero smoothing factor to be rejected")
		} catch (_: IllegalArgumentException) {}

		val logic = logic(smoothingFactor = 1.0)
		assertNotNull(logic)
	}

	@Test
	fun nonFiniteSensorVectorsAreIgnored() {
		val logic = logic(smoothingFactor = 1.0)
		logic.onGyroscopeSample(Float.NaN, 0f, 0f, timestampNanos = 0L)
		logic.onLinearAccelerationSample(Float.POSITIVE_INFINITY, 0f, 0f, timestampNanos = 0L)

		assertNull(logic.score(0L))
	}

	@Test
	fun duplicateAndBackwardsTimestampsDoNotAddEmaSteps() {
		val logic = logic(smoothingFactor = 0.5, gyroFullScale = 1.0)
		logic.onGyroscopeSample(0f, 0f, 0f, timestampNanos = 0L)
		logic.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 100L)
		logic.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 100L)
		logic.onGyroscopeSample(0f, 0f, 0f, timestampNanos = 99L)

		assertEquals(0.5, logic.score(100L)!!, 0.0)
	}

	@Test
	fun resetDropsSamplesAndSmoothing() {
		val logic = logic(smoothingFactor = 1.0)

		logic.onGyroscopeSample(1f, 0f, 0f, timestampNanos = 0L)
		assertEquals(0.5, logic.score(0L)!!, 0.0)
		logic.reset()

		assertNull(logic.score(0L))
	}

	private fun logic(
		staleTimeout: kotlin.time.Duration = 1_000.milliseconds,
		smoothingFactor: Double = 1.0,
		gyroFullScale: Double = 2.0,
		accelerationFullScale: Double = 4.0,
	): PhoneMotionScoreLogic = PhoneMotionScoreLogic(
		PhoneMotionMonitorConfig(
			staleTimeout = staleTimeout,
			smoothingFactor = smoothingFactor,
			gyroFullScaleRadPerSecond = gyroFullScale,
			linearAccelerationFullScaleMetersPerSecondSquared = accelerationFullScale,
			linearAccelerationWeight = 0.25,
		)
	)
}
