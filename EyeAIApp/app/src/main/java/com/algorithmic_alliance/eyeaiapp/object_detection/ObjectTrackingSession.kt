package com.algorithmic_alliance.eyeaiapp.object_detection

import com.algorithmic_alliance.eyeaiapp.camera.TrackingEpoch

internal class ObjectTrackingSession {
    private val modelLock = Any()
    private var trackerEpoch: TrackingEpoch? = null

    fun <T> withModelLock(block: () -> T): T = synchronized(modelLock) { block() }

    fun modelReplaced() = withModelLock { trackerEpoch = null }

    fun <T> run(
        epoch: TrackingEpoch,
        ready: () -> Boolean,
        admit: () -> Boolean,
        reset: () -> Unit,
        operation: () -> T,
    ): T? = withModelLock {
        if (!ready() || !admit()) return@withModelLock null
        if (trackerEpoch != epoch) {
            reset()
            trackerEpoch = epoch
        }
        operation()
    }
}
