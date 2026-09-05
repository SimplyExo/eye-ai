package com.algorithmic_alliance.eyeaiapp.camera

import com.algorithmic_alliance.eyeaiapp.NativeLib
import org.junit.Assert.*
import org.junit.Test
import uniffi.NativeLib.UniffiDetectedObject

class AnalysisResultsTest {
    private val generation = AnalysisGeneration(1, 2, 3, 4)
    private val objects = listOf(UniffiDetectedObject(0f, 0f, 1f, 1f, .5f, .5f, 1f, 1f, 1f, 0, "person", 1))
    private fun detection(arrival: Long = 0L) = ObjectDetectionSnapshot(objects, arrival, arrival, arrival, 1, generation)
    private fun depth(arrival: Long = 0L) = DepthSnapshot(NativeLib.NativeFloatBuffer(256 * 256), 256, 256, arrival, arrival, generation)

    @Test fun freshEmptyResultDiffersFromInvalidResult() {
        val snapshot = detection().copy(objects = emptyList())
        assertNotNull(AnalysisResults(generation, snapshot).freshObjects(0))
        assertTrue(AnalysisResults(generation, snapshot).freshObjects(0)!!.objects.isEmpty())
        assertNull(AnalysisResults(generation).freshObjects(0))
    }

    @Test fun ttlUsesFrameAgeEvenWhenCompletionIsRecent() {
        val snapshot = detection().copy(completedNanos = 1_100_000_000)
        assertNull(AnalysisResults(generation, snapshot).freshObjects(1_100_000_000))
        assertNull(AnalysisResults(generation, detection(100)).freshObjects(99))
    }

    @Test fun wrongRunSourceContentOrOdGenerationSuppressObjects() {
        for (other in listOf(generation.copy(run = 2), generation.copy(source = 4),
            generation.copy(content = 5), generation.copy(objectDetection = 3))) {
            assertNull(AnalysisResults(other, detection()).freshObjects(0))
        }
    }

    @Test fun depthIgnoresOdToggleButCannotCrossSourceOrRun() {
        assertNotNull(AnalysisResults(generation.copy(objectDetection = 3), depth = depth()).freshDepth(0))
        assertNull(AnalysisResults(generation.copy(source = 4), depth = depth()).freshDepth(0))
        assertNull(AnalysisResults(generation.copy(run = 2), depth = depth()).freshDepth(0))
    }

    @Test fun spatialSelectionSuppressesStaleOrMisalignedBoundingBoxes() {
        val results = AnalysisResults(generation, detection(), depth(250_000_000))
        assertEquals(objects, results.alignedObjects(250_000_000))
        assertTrue(results.copy(depth = depth(250_000_001)).alignedObjects(250_000_001).isEmpty())
        assertTrue(results.alignedObjects(1_000_000_001).isEmpty())
        assertTrue(results.copy(depth = null).alignedObjects(0).isEmpty())
    }
}
