package com.algorithmic_alliance.eyeaiapp.UI

import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import org.checkerframework.checker.guieffect.qual.UI


sealed interface UIEvent {
    data object VoskListeningChanged : UIEvent

    data object UpdateVoskStatusText : UIEvent

    data object InitVoskService : UIEvent

    data object CloseVoskService : UIEvent

    data object UpdateLlmStatusText: UIEvent

    data object UpdateSettings : UIEvent

    data object OnOpenSettings: UIEvent

    data object OnReturnFromSettings: UIEvent

    data class UIinitCamera(
        val previewView: PreviewView?,
        val lifecycleOwner: LifecycleOwner
    ) : UIEvent
}