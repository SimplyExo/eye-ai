package com.algorithmic_alliance.eyeaiapp.UI

import android.graphics.Bitmap
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import uniffi.NativeLib.UniffiDetectedObject

data class UIState(
    val reloadSettingsPageKey: Int = 0,
    val appMissingVoskPermission: Boolean = false,
    val speechRecognitionFinalResultText: String = "",
    val speechRecognitionPartialResultText: String = "",
    val llmResponseText: String = "",
    val depthPreviewBitmap: Bitmap? = null,
    val mediaPreviewBitmap: Bitmap? = null,
    val debugInputPreviewBitmap: Bitmap? = null,
    val performanceText: String = "",
    val detectedObjects: Array<UniffiDetectedObject> = emptyArray(),
    val cameraResolution: Size = Size(720, 1280),
    val ocrResults: Array<TextBoundingBox> = emptyArray(),
)