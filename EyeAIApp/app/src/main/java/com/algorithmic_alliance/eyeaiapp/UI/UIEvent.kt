package com.algorithmic_alliance.eyeaiapp.UI

import org.checkerframework.checker.guieffect.qual.UI

sealed interface UIEvent {
    data object VoskListeningChanged : UIEvent

    data object UpdateVoskStatusText: UIEvent

    data object InitVoskService: UIEvent

    data object CloseVoskService: UIEvent
}