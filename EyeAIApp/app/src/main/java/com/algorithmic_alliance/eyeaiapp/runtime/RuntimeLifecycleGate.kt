package com.algorithmic_alliance.eyeaiapp.runtime

/** Small, synchronized gate used to make runtime start/stop idempotent. */
internal class RuntimeLifecycleGate {
    @Volatile
    var isActive: Boolean = false
        private set

    @Synchronized
    fun start(): Boolean {
        if (isActive) return false
        isActive = true
        return true
    }

    @Synchronized
    fun stop(): Boolean {
        if (!isActive) return false
        isActive = false
        return true
    }
}
