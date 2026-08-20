package com.algorithmic_alliance.eyeaiapp.UI

data class UIState(
    val speechRecognitionFinalResultText: String = "",
    val speechRecognitionPartialResultText: String = "",
    val llmResponseText: String = ""
)