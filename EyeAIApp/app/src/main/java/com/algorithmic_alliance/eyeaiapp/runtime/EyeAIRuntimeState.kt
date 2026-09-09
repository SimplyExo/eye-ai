package com.algorithmic_alliance.eyeaiapp.runtime

import android.graphics.Bitmap
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.camera.FrameAnalysisUpdate
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
    val detectedObjects: Array<UniffiDetectedObject> = emptyArray(),
    val cameraResolution: Size = Size(720, 1280),
    val ocrResults: Array<TextBoundingBox> = emptyArray(),
    val lastError: String? = null,
)

internal fun EyeAIRuntimeState.withAnalysis(update: FrameAnalysisUpdate): EyeAIRuntimeState = copy(
    depthPreviewBitmap = update.depthPreviewBitmap ?: depthPreviewBitmap,
    debugInputPreviewBitmap = update.debugInputBitmap ?: debugInputPreviewBitmap,
    performanceText = update.performanceText ?: performanceText,
    detectedObjects = update.detectedObjects ?: detectedObjects,
    cameraResolution = update.frameSize ?: cameraResolution,
)
