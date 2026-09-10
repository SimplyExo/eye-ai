package com.algorithmic_alliance.eyeaiapp.inference.throttling

import com.algorithmic_alliance.eyeaiapp.Settings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ObjectDetectionPolicyTest {
	@Test
	fun frameRateSettingsUseOneToSixtyAndNullRemainsUnbounded() {
		assertEquals(1, Settings.normalizeDepthFrameRate(0))
		assertEquals(12, Settings.normalizeDepthFrameRate(12))
		assertEquals(60, Settings.normalizeDepthFrameRate(120))
		assertEquals(12, Settings.effectiveDepthFrameRate(12))
		assertNull(Settings.effectiveDepthFrameRate(null))
		assertEquals(1, Settings.normalizeObjectDetectionFrameRate(0))
		assertEquals(12, Settings.normalizeObjectDetectionFrameRate(12))
		assertEquals(60, Settings.normalizeObjectDetectionFrameRate(120))
	}

	@Test
	fun representativeBudgetsPreserveTheExactRelativePolicy() {
		assertRates(1.0, 1.0, 1.0, 1.0)
		assertRates(2.0, 1.0, 1.3, 2.0)
		assertRates(6.0, 2.1, 3.9, 6.0)
		assertRates(10.0, 3.5, 6.5, 10.0)
		assertRates(15.0, 5.25, 9.75, 15.0)
		assertRates(20.0, 7.0, 13.0, 20.0)
		assertRates(30.0, 10.5, 19.5, 30.0)
		assertRates(60.0, 21.0, 39.0, 60.0)
	}

	@Test
	fun enabledValuesClampToTheSettingsSliderRange() {
		assertRates(0.5, 1.0, 1.0, 1.0)
		assertRates(120.0, 21.0, 39.0, 60.0)
		assertEquals(1, Settings.normalizeObjectDetectionFrameRate(0))
		assertEquals(60, Settings.normalizeObjectDetectionFrameRate(120))
	}

	@Test
	fun disabledLimiterHasNoInterval() {
		val budget = ObjectDetectionPolicy.budget(null)
		assertNull(budget.maxFps)
		assertEquals(0L, budget.quietIntervalNanos)
		assertEquals(0L, budget.activeIntervalNanos)
		assertEquals(0L, budget.burstIntervalNanos)
	}

	private fun assertRates(user: Double, quiet: Double, active: Double, burst: Double) {
		val rates = ObjectDetectionPolicy.targetRates(user)
		assertEquals(quiet, rates.quietFps, 0.0001)
		assertEquals(active, rates.activeFps, 0.0001)
		assertEquals(burst, rates.burstFps, 0.0001)
		val budget = ObjectDetectionPolicy.budget(user)
		assertTrue(budget.quietIntervalNanos >= budget.activeIntervalNanos)
		assertTrue(budget.activeIntervalNanos >= budget.burstIntervalNanos)
		assertTrue(1e9 / budget.quietIntervalNanos <= quiet + 0.0001)
		assertTrue(1e9 / budget.activeIntervalNanos <= active + 0.0001)
		assertTrue(1e9 / budget.burstIntervalNanos <= burst + 0.0001)
	}
}
