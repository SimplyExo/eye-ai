package com.algorithmic_alliance.eyeaiapp.inference

import org.junit.Assert.*
import org.junit.Test

class ObjectDetectionV1PolicyTest {
    @Test fun technicalProfileHasOrderedThreeSevenFifteenHertzIntervals() {
        val config = ObjectDetectionV1Policy.config()
        assertEquals(3.0, 1e9 / config.quietInterval.inWholeNanoseconds, 0.0001)
        assertEquals(7.0, 1e9 / config.activeInterval.inWholeNanoseconds, 0.0001)
        assertEquals(15.0, 1e9 / config.burstInterval.inWholeNanoseconds, 0.0001)
    }

    @Test fun directModeChangesAndLowUserCapNeverCreateCatchUpSlots() {
        val scheduler = InferenceScheduler(ObjectDetectionV1Policy.config(1.0), MonotonicClock { 0L })
        assertTrue(scheduler.tryAcquireInference(0))
        assertEquals(InferenceMode.ACTIVE, scheduler.updatePhoneMotion(1.0, 100_000_000).mode)
        assertFalse(scheduler.tryAcquireInference(100_000_000))
        assertEquals(InferenceMode.BURST, scheduler.updateVisualChange(1.0, 200_000_000).mode)
        assertFalse(scheduler.tryAcquireInference(200_000_000))
        assertTrue(scheduler.tryAcquireInference(1_000_000_000))
        assertTrue(scheduler.tryAcquireInference(10_000_000_000))
        assertFalse(scheduler.tryAcquireInference(10_000_000_000))
    }

    @Test fun resettingEvidenceDoesNotBypassTheCap() {
        var now = 0L
        val scheduler = InferenceScheduler(ObjectDetectionV1Policy.config(2.0), MonotonicClock { now })
        assertTrue(scheduler.tryAcquireInference())
        now = 100_000_000
        scheduler.resetActivity()
        scheduler.updateVisualChange(1.0)
        assertFalse(scheduler.tryAcquireInference())
        now = 500_000_000
        assertTrue(scheduler.tryAcquireInference())
    }
}
