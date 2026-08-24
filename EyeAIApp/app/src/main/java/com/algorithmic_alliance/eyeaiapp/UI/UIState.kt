package com.algorithmic_alliance.eyeaiapp.UI

import android.graphics.Bitmap

data class UIState(
    val speechRecognitionFinalResultText: String = "",
    val speechRecognitionPartialResultText: String = "",
    val llmResponseText: String = "",
    val depthPreviewBitmap: Bitmap? = null
)