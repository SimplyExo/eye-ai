package com.algorithmic_alliance.eyeaiapp.UI

import android.graphics.Bitmap

data class UIState(
    val speechRecognitionFinalResultText: String = "",
    val speechRecognitionPartialResultText: String = "",
    val llmResponseText: String = "",
    val depthPreviewBitmap: Bitmap? = null,
    val mediaPreviewBitmap: Bitmap? = null,
    val debugInputPreviewBitmap: Bitmap? = null,
    val mediaPreviewVisible: Boolean = false,
    val cameraPreviewVisible: Boolean = true,
    val debugInputPreviewVisible: Boolean = false
)