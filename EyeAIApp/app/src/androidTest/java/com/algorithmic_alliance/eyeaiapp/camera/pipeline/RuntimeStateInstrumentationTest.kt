package com.algorithmic_alliance.eyeaiapp.camera

import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntimeState
import com.algorithmic_alliance.eyeaiapp.runtime.withAnalysis
import org.junit.Assert.*
import org.junit.Test
import uniffi.NativeLib.UniffiDetectedObject

class RuntimeStateInstrumentationTest {
    @Test fun runtimeStateDistinguishesNoUpdateEmptyAndInvalidation() {
        val generation = AnalysisGeneration(1, 2, 3, 4)
        val objects = listOf(UniffiDetectedObject(0f, 0f, 1f, 1f, .5f, .5f, 1f, 1f, 1f, 0, "person", 1))
        val snapshot = ObjectDetectionSnapshot(objects, 0, 0, 0, 1, generation)
        val depth = DepthSnapshot(NativeLib.NativeFloatBuffer(1), 1, 1, 0, 0, generation)
        val results = AnalysisResults(generation, snapshot, depth)
        val preview = pipelinePixels()
        val state = EyeAIRuntimeState(analysisResults = results, depthPreviewBitmap = preview)
        assertSame(results, state.withAnalysis(FrameAnalysisUpdate()).analysisResults)
        val empty = state.withAnalysis(FrameAnalysisUpdate(results = results.copy(objects = snapshot.copy(objects = emptyList()))))
        assertNotNull(empty.analysisResults.objects)
        assertTrue(empty.detectedObjects.isEmpty())
        assertSame(preview, empty.depthPreviewBitmap)
        val invalid = state.withAnalysis(FrameAnalysisUpdate(results = AnalysisResults(generation.copy(run = 2))))
        assertNull(invalid.analysisResults.objects)
        assertNull(invalid.depthPreviewBitmap)
        assertTrue(invalid.detectedObjects.isEmpty())
    }
}
