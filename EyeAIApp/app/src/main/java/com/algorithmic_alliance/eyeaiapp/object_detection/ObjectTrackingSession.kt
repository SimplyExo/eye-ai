package com.algorithmic_alliance.eyeaiapp.object_detection

import com.algorithmic_alliance.eyeaiapp.camera.TrackingEpoch

/**
 * The YOLO model's serialization boundary, shared by model creation and inference.
 * Admission may briefly acquire Analyzer.stateLock; callers must never enter this
 * session while holding that lock. Native calls happen only after admission returns.
 *
 * Reset is lazy: an in-flight old operation finishes before the first admitted new
 * epoch resets the tracker. No old operation can finish into the new tracker state.
 */
internal class ObjectTrackingSession {
    private val modelLock = Any()
    private var trackerEpoch: TrackingEpoch? = null

    fun <T> withModelLock(block: () -> T): T = synchronized(modelLock) { block() }

    /** Called after native model/tracker replacement under the same model lock. */
    fun modelReplaced() = withModelLock { trackerEpoch = null }

    fun <T> run(
        epoch: TrackingEpoch,
        ready: () -> Boolean,
        admit: () -> Boolean,
        reset: () -> Unit,
        operation: () -> T,
    ): T? = withModelLock {
        // A stale/disabled/skipped request must not reset or mutate any tracker.
        if (!ready() || !admit()) return@withModelLock null
        if (trackerEpoch != epoch) {
            reset()
            // Only acknowledge a successful reset; a failure is retried next time.
            trackerEpoch = epoch
        }
        operation()
    }
}
