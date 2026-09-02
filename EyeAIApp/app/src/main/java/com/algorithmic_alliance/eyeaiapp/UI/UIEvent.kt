package com.algorithmic_alliance.eyeaiapp.UI

import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner


sealed interface UIEvent {
    data object VoskListeningChanged : UIEvent

    data object UpdateVoskStatusText : UIEvent

    data object InitVoskService : UIEvent

    data object CloseVoskService : UIEvent

    data object UpdateSpeechStatusText : UIEvent

    data object UpdateSettings : UIEvent

    data object OnOpenSettings : UIEvent

    data object OnReloadSettingsPage: UIEvent

    data object OnReloadDebugPage: UIEvent

    data object OnReturnFromSettings : UIEvent

    data class OnUpdatePermissionTutorialCompleted(val value: Boolean): UIEvent

    data class OnUpdateConnectionTutorialCompleted(val value: Boolean): UIEvent

    data class OnUpdateAppMissingVoskPermission(val value: Boolean) : UIEvent
    data class OnUpdateAppMissingCameraPermission(val value: Boolean) : UIEvent

    data class OnUpdateActionStartedFromSettings(val value: Boolean): UIEvent

    data class OnUpdateAppMissingSelectedMediaSource(val value: Boolean) : UIEvent

    data class OnUpdateTTSSpeaking(val value: Boolean): UIEvent

    data class OnUpdateSettingsOpened(val value: Boolean): UIEvent

    data class UIinitCamera(
        val previewView: PreviewView?,
        val lifecycleOwner: LifecycleOwner
    ) : UIEvent
}