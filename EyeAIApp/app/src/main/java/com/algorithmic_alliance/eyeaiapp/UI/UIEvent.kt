package com.algorithmic_alliance.eyeaiapp.UI

import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner


sealed interface UIEvent {
    data object VoskListeningChanged : UIEvent

    data object UpdateVoskStatusText: UIEvent

    data object InitVoskService: UIEvent

    data object CloseVoskService: UIEvent

    data class CameraPreviewReady(val previewView: PreviewView, val lifecycleOwner: LifecycleOwner): UIEvent
}