package com.algorithmic_alliance.eyeaiapp.inference

import com.algorithmic_alliance.eyeaiapp.Settings
import org.junit.Assert.*
import org.junit.Test

class ObjectDetectionV1PolicyTest {
    @Test
    fun depthLimiterUsesOneToSixtyAndNullRemainsUnbounded() {
        assertEquals(1, Settings.normalizeDepthFrameRate(0))
        assertEquals(12, Settings.normalizeDepthFrameRate(12))
        assertEquals(60, Settings.normalizeDepthFrameRate(120))
        assertEquals(12, Settings.effectiveDepthFrameRate(12))
        assertNull(Settings.effectiveDepthFrameRate(null))
    }

    @Test fun budgetFiveUsesFiveFpsForEveryMode() {
        assertRates(5.0, 5.0, 5.0, 5.0)
    }

    @Test fun budgetEightUsesFiveFivePointTwoAndEightFps() {
        assertRates(8.0, 5.0, 5.2, 8.0)
    }

    @Test fun budgetTenUsesFiveSixPointFiveAndTenFps() {
        assertRates(10.0, 5.0, 6.5, 10.0)
    }

    @Test fun budgetFifteenUsesFivePointTwoFiveNinePointSevenFiveAndFifteenFps() {
        assertRates(15.0, 5.25, 9.75, 15.0)
    }

    @Test fun budgetTwentyUsesSevenThirteenAndTwentyFps() {
        assertRates(20.0, 7.0, 13.0, 20.0)
    }

    @Test fun budgetThirtyUsesTenPointFiveNineteenPointFiveAndThirtyFps() {
        assertRates(30.0, 10.5, 19.5, 30.0)
    }

    @Test fun budgetSixtyUsesTwentyOneThirtyNineAndSixtyFps() {
        assertRates(60.0, 21.0, 39.0, 60.0)
    }

    @Test fun valuesAboveTheSliderMaximumKeepTheEffectiveBudgetAtSixty() {
        assertRates(120.0, 21.0, 39.0, 60.0)
        assertEquals(
            60.0,
            checkNotNull(ObjectDetectionV1Policy.config(120.0).maxObjectDetectionRateHz),
            0.0,
        )
        assertEquals(60, Settings.normalizeObjectDetectionFrameRate(120))
    }

    @Test fun disabledObjectLimiterAddsNoArtificialCadenceDelay() {
        val config = ObjectDetectionV1Policy.config(null)
        assertNull(config.maxObjectDetectionRateHz)
        assertEquals(1L, config.quietInterval.inWholeNanoseconds)
        assertEquals(1L, config.activeInterval.inWholeNanoseconds)
        assertEquals(1L, config.burstInterval.inWholeNanoseconds)

        val scheduler = InferenceScheduler(config, MonotonicClock { 0L })
        assertTrue(scheduler.tryAcquireInference(0L))
        assertFalse(scheduler.tryAcquireInference(0L))
        assertTrue(scheduler.tryAcquireInference(1L))
    }

    @Test fun valuesBelowTheUserFloorAreSafelyRaisedToFive() {
        assertRates(1.0, 5.0, 5.0, 5.0)
        assertEquals(5, com.algorithmic_alliance.eyeaiapp.Settings.normalizeObjectDetectionFrameRate(1))
    }

    private fun assertRates(user: Double, quiet: Double, active: Double, burst: Double) {
        val rates = ObjectDetectionV1Policy.targetRates(user)
        assertEquals(quiet, rates.quietFps, 0.0001)
        assertEquals(active, rates.activeFps, 0.0001)
        assertEquals(burst, rates.burstFps, 0.0001)
        assertEquals(
            rates.budgetFps,
            minOf(
                user.coerceAtLeast(ObjectDetectionV1Policy.MIN_USER_MAX_FPS),
                ObjectDetectionV1Policy.MAX_USER_MAX_FPS,
            ),
            0.0001,
        )
        val config = ObjectDetectionV1Policy.config(user)
        assertTrue(config.quietInterval >= config.activeInterval)
        assertTrue(config.activeInterval >= config.burstInterval)
        assertTrue(1e9 / config.quietInterval.inWholeNanoseconds <= quiet + 0.0001)
        assertTrue(1e9 / config.activeInterval.inWholeNanoseconds <= active + 0.0001)
        assertTrue(1e9 / config.burstInterval.inWholeNanoseconds <= burst + 0.0001)
    }

    @Test fun directModeChangesAtFiveFpsNeverCreateCatchUpSlots() {
        val scheduler = InferenceScheduler(ObjectDetectionV1Policy.config(5.0), MonotonicClock { 0L })
        assertTrue(scheduler.tryAcquireInference(0))
        assertEquals(InferenceMode.ACTIVE, scheduler.updatePhoneMotion(1.0, 100_000_000).mode)
        assertFalse(scheduler.tryAcquireInference(100_000_000))
        assertEquals(InferenceMode.BURST, scheduler.updateVisualChange(1.0, 200_000_000).mode)
        assertTrue(scheduler.tryAcquireInference(200_000_000))
        assertFalse(scheduler.tryAcquireInference(200_000_000))
        assertTrue(scheduler.tryAcquireInference(400_000_000))
        assertTrue(scheduler.tryAcquireInference(1_000_000_000))
        assertTrue(scheduler.tryAcquireInference(10_000_000_000))
        assertFalse(scheduler.tryAcquireInference(10_000_000_000))
    }

    @Test fun resettingEvidenceDoesNotBypassTheCap() {
        var now = 0L
        val scheduler = InferenceScheduler(ObjectDetectionV1Policy.config(5.0), MonotonicClock { now })
        assertTrue(scheduler.tryAcquireInference())
        now = 100_000_000
        scheduler.resetActivity()
        scheduler.updateVisualChange(1.0)
        assertFalse(scheduler.tryAcquireInference())
        now = 500_000_000
        assertTrue(scheduler.tryAcquireInference())
    }
}
