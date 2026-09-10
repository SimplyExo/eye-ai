package com.algorithmic_alliance.eyeaiapp.inference.throttling

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SceneChangeMonitorTest {
	@Test
	fun firstFrameIsACoherentBaselineSample() {
		val sample = monitor().sampleLuma(luma(4, 4, 10), 4, 4, nowNanos = 0L)
		assertNotNull(sample)
		assertTrue(checkNotNull(sample).baselineFrame)
		assertEquals(0.0, sample.score, 0.0)
	}

	@Test
	fun identicalCompleteAndPartialChangesScoreAsExpected() {
		val unchanged = monitor().run {
			sampleLuma(luma(4, 4, 10), 4, 4, nowNanos = 0L)
			checkNotNull(sampleLuma(luma(4, 4, 10), 4, 4, nowNanos = 1L)).score
		}
		val complete = monitor().run {
			sampleLuma(luma(4, 4, 0), 4, 4, nowNanos = 0L)
			checkNotNull(sampleLuma(luma(4, 4, 255), 4, 4, nowNanos = 1L)).score
		}
		val partial = monitor().run {
			sampleLuma(luma(4, 4, 0), 4, 4, nowNanos = 0L)
			val next = luma(4, 4, 0).also { it.fill(255.toByte(), 0, 8) }
			checkNotNull(sampleLuma(next, 4, 4, nowNanos = 1L)).score
		}
		assertEquals(0.0, unchanged, 0.0)
		assertTrue(complete > 0.95)
		assertTrue(partial in 0.2..<complete)
	}

	@Test
	fun cadenceReturnsNullUntilTheNextSampleIsDue() {
		val monitor = monitor(cadenceNanos = 100L)
		assertNotNull(monitor.sampleLuma(luma(4, 4, 0), 4, 4, nowNanos = 0L))
		assertNull(monitor.sampleLuma(luma(4, 4, 255), 4, 4, nowNanos = 99L))
		assertTrue(checkNotNull(monitor.sampleLuma(luma(4, 4, 255), 4, 4, nowNanos = 100L)).score > 0.95)
	}

	@Test
	fun dimensionRotationAndBackwardsTimeStartANewBaseline() {
		fun changedSample(change: (SceneChangeMonitor) -> SceneSample?): SceneSample {
			val monitor = monitor()
			monitor.sampleLuma(luma(4, 4, 0), 4, 4, nowNanos = 10L)
			return checkNotNull(change(monitor))
		}
		val dimension = changedSample { it.sampleLuma(luma(2, 2, 255), 2, 2, nowNanos = 11L) }
		val rotation = changedSample {
			it.sampleLuma(luma(4, 4, 255), 4, 4, rotationDegrees = 90, nowNanos = 11L)
		}
		val backwards = changedSample { it.sampleLuma(luma(4, 4, 255), 4, 4, nowNanos = 9L) }
		for (sample in listOf(dimension, rotation, backwards)) {
			assertTrue(sample.baselineFrame)
			assertEquals(0.0, sample.score, 0.0)
		}
	}

	@Test
	fun resetClearsBaselineAndCadence() {
		val monitor = monitor(cadenceNanos = 1_000L)
		monitor.sampleLuma(luma(4, 4, 0), 4, 4, nowNanos = 0L)
		monitor.reset()
		val sample = checkNotNull(monitor.sampleLuma(luma(4, 4, 255), 4, 4, nowNanos = 1L))
		assertTrue(sample.baselineFrame)
	}

	@Test
	fun sourceArrayAndRowPaddingAreNotRetainedOrScored() {
		val monitor = SceneChangeMonitor(SceneChangeConfig(2, 2, sampleCadenceNanos = 0L))
		val baseline = byteArrayOf(10, 10, 99, 99, 20, 20, 99, 99)
		monitor.sampleLuma(baseline, 2, 2, rowStride = 4, nowNanos = 0L)
		baseline.fill(0)
		val paddingChanged = byteArrayOf(
			10, 10, 255.toByte(), 255.toByte(),
			20, 20, 255.toByte(), 255.toByte(),
		)
		val sample = checkNotNull(
			monitor.sampleLuma(paddingChanged, 2, 2, rowStride = 4, nowNanos = 1L),
		)
		assertFalse(sample.baselineFrame)
		assertEquals(0.0, sample.score, 0.0)
	}

	@Test
	fun sameLumaSequenceIsDeterministic() {
		fun scores() = monitor().let { monitor ->
			listOf(10, 10, 80, 80, 240, 30).mapIndexed { index, value ->
				checkNotNull(monitor.sampleLuma(luma(4, 4, value), 4, 4, nowNanos = index.toLong())).score
			}
		}
		assertEquals(scores(), scores())
	}

	private fun monitor(cadenceNanos: Long = 0L) = SceneChangeMonitor(
		SceneChangeConfig(sampleWidth = 4, sampleHeight = 4, sampleCadenceNanos = cadenceNanos),
	)

	private fun luma(width: Int, height: Int, value: Int) =
		ByteArray(width * height) { value.toByte() }
}
