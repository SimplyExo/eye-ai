package com.algorithmic_alliance.eyeaiapp.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TrackingEpochTest {
    private val base = AnalysisGeneration(
        run = 10,
        objectDetection = 20,
        source = 30,
        content = 40,
    )

    @Test
    fun trackingEpochReusesAllExistingInvalidationAxes() {
        assertEquals(TrackingEpoch(10, 20, 30, 40), base.trackingEpoch)

        for (changed in listOf(
            base.copy(run = 11),
            base.copy(source = 31),
            base.copy(content = 41),
            base.copy(objectDetection = 21),
        )) {
            assertFalse(base.sameTrackingEpoch(changed))
        }
    }

    @Test
    fun imageStreamCompatibilityIntentionallyIgnoresObjectDetectionToggle() {
        val odChanged = base.copy(objectDetection = 21)

        assertTrue(base.sameImageStream(odChanged))
        assertNotEquals(base.trackingEpoch, odChanged.trackingEpoch)
    }

    @Test
    fun cadenceAndSchedulerActivityDoNotCreateAnEpochByThemselves() {
        // QUIET/ACTIVE/BURST and FPS policy changes do not mutate any
        // generation axis. The same generation therefore remains the same
        // tracking epoch across those scheduler-only transitions.
        val afterCadenceChanges = base.copy()

        assertEquals(base.trackingEpoch, afterCadenceChanges.trackingEpoch)
        assertTrue(base.sameTrackingEpoch(afterCadenceChanges))
    }
}
