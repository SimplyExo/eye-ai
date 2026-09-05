package com.algorithmic_alliance.eyeaiapp.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LumaSceneChangeScorerTest {
	@Test
	fun `first sample establishes baseline without a change`() {
		val scorer = scorer(sampleCount = 8)

		assertEquals(0.0, scorer.score(luma(8, 40)), 0.0)
		assertTrue(scorer.hasBaseline)
	}

	@Test
	fun `identical samples score zero`() {
		val scorer = scorer(sampleCount = 8)
		val frame = luma(8, 40)

		scorer.score(frame)

		assertEquals(0.0, scorer.score(frame), 0.0)
	}

	@Test
	fun `complete change scores high`() {
		val scorer = scorer(sampleCount = 16)

		scorer.score(luma(16, 0))

		assertTrue(scorer.score(luma(16, 255)) > 0.95)
	}

	@Test
	fun `partial change produces an intermediate score`() {
		val scorer = scorer(sampleCount = 16)

		scorer.score(luma(16, 0))
		val changed = luma(16, 0)
		changed.fill(255.toByte(), fromIndex = 0, toIndex = 8)

		val score = scorer.score(changed)

		assertTrue(score > 0.2)
		assertTrue(score < 0.8)
	}

	@Test
	fun `score stays in normalized range`() {
		val scorer = scorer(sampleCount = 32)

		scorer.score(luma(32, 0))
		val score = scorer.score(luma(32, 255))

		assertTrue(score in 0.0..1.0)
	}

	@Test
	fun `small luma noise is below the deadband`() {
		val scorer = scorer(sampleCount = 16)

		scorer.score(luma(16, 100))
		val noisy = ByteArray(16) { index -> (100 + (index % 2) * 4 - 2).toByte() }

		assertEquals(0.0, scorer.score(noisy), 0.0)
	}

	@Test
	fun `small consistent exposure shift is compensated`() {
		val scorer = scorer(sampleCount = 16)

		scorer.score(luma(16, 100))

		assertEquals(0.0, scorer.score(luma(16, 120)), 0.01)
	}

	@Test
	fun `reset makes the next sample baseline only`() {
		val scorer = scorer(sampleCount = 8)

		scorer.score(luma(8, 0))
		scorer.reset()

		assertFalse(scorer.hasBaseline)
		assertEquals(0.0, scorer.score(luma(8, 255)), 0.0)
	}

	@Test
	fun `input array is copied and not retained`() {
		val scorer = scorer(sampleCount = 8)
		val first = luma(8, 20)

		scorer.score(first)
		first.fill(240.toByte())

		assertEquals(0.0, scorer.score(luma(8, 20)), 0.0)
	}

	private fun scorer(sampleCount: Int): LumaSceneChangeScorer =
		LumaSceneChangeScorer(
			sampleCount = sampleCount,
			noiseFloor = 8.0 / 255.0,
			exposureCompensationLimit = 32.0 / 255.0,
			exposureConsistencyTolerance = 16.0 / 255.0,
		)

	private fun luma(size: Int, value: Int): ByteArray = ByteArray(size) { value.toByte() }
}
