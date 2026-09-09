package com.algorithmic_alliance.eyeaiapp.runtime

import android.annotation.SuppressLint
import android.content.Context
import android.os.PowerManager

/**
 * Owns the one CPU wake lock required by the local continuous camera mode.
 * It is intentionally not a global lock and is released by every stop path.
 */
internal class EyeAIWakeLock(context: Context) {
    private val wakeLock = (context.applicationContext
        .getSystemService(Context.POWER_SERVICE) as PowerManager)
        .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "EyeAI::ContinuousInference")
        .apply { setReferenceCounted(false) }

    @Synchronized
    @SuppressLint("WakelockTimeout")
    fun acquire() {
        if (!wakeLock.isHeld) wakeLock.acquire()
    }

    @Synchronized
    fun release() {
        if (wakeLock.isHeld) wakeLock.release()
    }

    val isHeld: Boolean
        @Synchronized get() = wakeLock.isHeld
}
