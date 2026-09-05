package com.algorithmic_alliance.eyeaiapp.inference

import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SceneChangeMonitorTest {
	@Test
	fun `first frame establishes baseline`() {
		val monitor = monitor()

		val result = monitor.updateLumaDetailed(luma(4, 4, 10), width = 4, height = 4, timestampNanos = 0L)

		assertTrue(result.sampled)
		assertTrue(result.baselineFrame)
		assertEquals(0.0, result.visualChangeScore, 0.0)
	}

	@Test
	fun `numeric update exposes the score without a result snapshot`() {
		val monitor = monitor()

		val firstScore = monitor.updateLuma(luma(4, 4, 0), width = 4, height = 4, timestampNanos = 0L)
		val secondScore = monitor.updateLuma(luma(4, 4, 255), width = 4, height = 4, timestampNanos = 1L)

		assertEquals(0.0, firstScore, 0.0)
		assertTrue(secondScore > 0.95)
		assertTrue(monitor.lastCallWasSampled)
		assertFalse(monitor.lastCallWasBaselineFrame)
	}

	@Test
	fun `identical frames score zero`() {
		val monitor = monitor()

		monitor.updateLumaDetailed(luma(4, 4, 10), width = 4, height = 4, timestampNanos = 0L)
		val result = monitor.updateLumaDetailed(luma(4, 4, 10), width = 4, height = 4, timestampNanos = 1L)

		assertTrue(result.sampled)
		assertEquals(0.0, result.visualChangeScore, 0.0)
	}

	@Test
	fun `complete luma change is high and partial change is intermediate`() {
		val monitor = monitor()

		monitor.updateLumaDetailed(luma(4, 4, 0), width = 4, height = 4, timestampNanos = 0L)
		val complete = monitor.updateLumaDetailed(luma(4, 4, 255), width = 4, height = 4, timestampNanos = 1L)

		monitor.reset()
		monitor.updateLumaDetailed(luma(4, 4, 0), width = 4, height = 4, timestampNanos = 2L)
		val partialFrame = luma(4, 4, 0)
		partialFrame.fill(255.toByte(), fromIndex = 0, toIndex = 8)
		val partial = monitor.updateLumaDetailed(partialFrame, width = 4, height = 4, timestampNanos = 3L)

		assertTrue(complete.visualChangeScore > 0.95)
		assertTrue(partial.visualChangeScore > 0.2)
		assertTrue(partial.visualChangeScore < complete.visualChangeScore)
	}

	@Test
	fun `sample cadence returns last score until next due sample`() {
		val monitor = SceneChangeMonitor(
			SceneChangeMonitorConfig(
				sampleWidth = 4,
				sampleHeight = 4,
				sampleCadence = 100.milliseconds,
			)
		)

		monitor.updateLumaDetailed(luma(4, 4, 0), width = 4, height = 4, timestampNanos = 0L)
		val skipped = monitor.updateLumaDetailed(
			luma(4, 4, 255),
			width = 4,
			height = 4,
			timestampNanos = 50.milliseconds.inWholeNanoseconds,
		)
		val sampled = monitor.updateLumaDetailed(
			luma(4, 4, 255),
			width = 4,
			height = 4,
			timestampNanos = 100.milliseconds.inWholeNanoseconds,
		)

		assertFalse(skipped.sampled)
		assertEquals(0.0, skipped.visualChangeScore, 0.0)
		assertTrue(sampled.sampled)
		assertTrue(sampled.visualChangeScore > 0.95)
	}

	@Test
	fun `dimension change establishes a new baseline`() {
		val monitor = monitor()

		monitor.updateLumaDetailed(luma(4, 4, 0), width = 4, height = 4, timestampNanos = 0L)
		monitor.updateLumaDetailed(luma(4, 4, 255), width = 4, height = 4, timestampNanos = 1L)
		val result = monitor.updateLumaDetailed(luma(2, 2, 255), width = 2, height = 2, timestampNanos = 2L)

		assertTrue(result.sampled)
		assertTrue(result.baselineFrame)
		assertEquals(0.0, result.visualChangeScore, 0.0)
	}

	@Test
	fun `rotation change establishes a new baseline`() {
		val monitor = monitor()

		monitor.updateLumaDetailed(
			luma(4, 4, 0),
			width = 4,
			height = 4,
			rotationDegrees = 0,
			timestampNanos = 0L,
		)
		val result = monitor.updateLumaDetailed(
			luma(4, 4, 255),
			width = 4,
			height = 4,
			rotationDegrees = 90,
			timestampNanos = 1L,
		)

		assertTrue(result.baselineFrame)
		assertEquals(0.0, result.visualChangeScore, 0.0)
	}

	@Test
	fun `reset clears baseline and cadence`() {
		val monitor = SceneChangeMonitor(
			SceneChangeMonitorConfig(
				sampleWidth = 4,
				sampleHeight = 4,
				sampleCadence = 1_000.milliseconds,
			)
		)

		monitor.updateLumaDetailed(luma(4, 4, 0), width = 4, height = 4, timestampNanos = 0L)
		monitor.reset()
		val result = monitor.updateLumaDetailed(luma(4, 4, 255), width = 4, height = 4, timestampNanos = 1L)

		assertTrue(result.baselineFrame)
		assertEquals(0.0, result.visualChangeScore, 0.0)
	}

	@Test
	fun `out of order timestamp resets baseline instead of comparing stale frames`() {
		val monitor = monitor()

		monitor.updateLumaDetailed(luma(4, 4, 0), width = 4, height = 4, timestampNanos = 10L)
		val result = monitor.updateLumaDetailed(luma(4, 4, 255), width = 4, height = 4, timestampNanos = 9L)

		assertTrue(result.baselineFrame)
		assertEquals(0.0, result.visualChangeScore, 0.0)
	}

	@Test
	fun `source luma array is not retained`() {
		val monitor = monitor()
		val first = luma(4, 4, 20)

		monitor.updateLumaDetailed(first, width = 4, height = 4, timestampNanos = 0L)
		first.fill(240.toByte())
		val result = monitor.updateLumaDetailed(luma(4, 4, 20), width = 4, height = 4, timestampNanos = 1L)

		assertEquals(0.0, result.visualChangeScore, 0.0)
	}

	@Test
	fun `strided luma plane ignores row padding`() {
		val monitor = SceneChangeMonitor(
			SceneChangeMonitorConfig(
				sampleWidth = 2,
				sampleHeight = 2,
				sampleCadence = Duration.ZERO,
			)
		)
		val baseline = byteArrayOf(
			10, 10, 99, 99,
			20, 20, 99, 99,
		)
		val paddingOnlyChange = byteArrayOf(
			10, 10, 255.toByte(), 255.toByte(),
			20, 20, 255.toByte(), 255.toByte(),
		)

		monitor.updateLumaDetailed(baseline, width = 2, height = 2, rowStride = 4, timestampNanos = 0L)
		val result = monitor.updateLumaDetailed(
			paddingOnlyChange,
			width = 2,
			height = 2,
			rowStride = 4,
			timestampNanos = 1L,
		)

		assertEquals(0.0, result.visualChangeScore, 0.0)
	}

	@Test
	fun `same sequence is deterministic`() {
		val first = monitor()
		val second = monitor()
		val frames = listOf(10, 10, 80, 80, 240, 30)

		val firstScores = frames.mapIndexed { index, value ->
			first.updateLumaDetailed(
				luma(4, 4, value),
				width = 4,
				height = 4,
				timestampNanos = index.toLong(),
			).visualChangeScore
		}
		val secondScores = frames.mapIndexed { index, value ->
			second.updateLumaDetailed(
				luma(4, 4, value),
				width = 4,
				height = 4,
				timestampNanos = index.toLong(),
			).visualChangeScore
		}

		assertEquals(firstScores, secondScores)
	}

	private fun monitor(): SceneChangeMonitor =
		SceneChangeMonitor(
			SceneChangeMonitorConfig(
				sampleWidth = 4,
				sampleHeight = 4,
				sampleCadence = Duration.ZERO,
			)
		)

	private fun luma(width: Int, height: Int, value: Int): ByteArray =
		ByteArray(width * height) { value.toByte() }
}
