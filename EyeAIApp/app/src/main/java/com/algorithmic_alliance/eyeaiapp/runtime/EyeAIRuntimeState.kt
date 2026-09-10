package com.algorithmic_alliance.eyeaiapp.runtime

import android.graphics.Bitmap
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.camera.FrameAnalysisUpdate
import com.algorithmic_alliance.eyeaiapp.camera.AnalysisResults
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import uniffi.NativeLib.UniffiDetectedObject

/** Immutable state exposed to short-lived UI observers. */
data class EyeAIRuntimeState(
    val operationActive: Boolean = false,
    val cameraActive: Boolean = false,
    val voskListening: Boolean = false,
    val ttsSpeaking: Boolean = false,
    val speechRecognitionFinalResultText: String = "",
    val speechRecognitionPartialResultText: String = "",
    val speechResponseText: String = "",
    val depthPreviewBitmap: Bitmap? = null,
    val debugInputPreviewBitmap: Bitmap? = null,
    val mediaPreviewBitmap: Bitmap? = null,
    val performanceText: String = "",
    val analysisResults: AnalysisResults = AnalysisResults(),
    val cameraResolution: Size = Size(720, 1280),
    val ocrResults: Array<TextBoundingBox> = emptyArray(),
    val lastError: String? = null,
) {
    val detectedObjects: Array<UniffiDetectedObject>
        get() = analysisResults.objects?.objects?.toTypedArray() ?: emptyArray()
}

internal fun EyeAIRuntimeState.withAnalysis(update: FrameAnalysisUpdate): EyeAIRuntimeState = copy(
    depthPreviewBitmap = if (update.results != null && update.results.depth == null) null
        else update.depthPreviewBitmap ?: depthPreviewBitmap,
    debugInputPreviewBitmap = if (update.results != null && update.results.depth == null) null
        else update.debugInputBitmap ?: debugInputPreviewBitmap,
    performanceText = update.performanceText ?: performanceText,
    analysisResults = update.results ?: analysisResults,
    cameraResolution = update.frameSize ?: cameraResolution,
)
