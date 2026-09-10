package com.algorithmic_alliance.eyeaiapp.inference.throttling.motion

class PhoneMotionLifecycle(private val createMonitor: () -> PhoneMotionMonitor) {
    private var monitor: PhoneMotionMonitor? = null
    private var requested = false

    @Synchronized
    fun update(
        operationActive: Boolean,
        objectDetectionEnabled: Boolean,
        limiterEnabled: Boolean,
        profilingEnabled: Boolean,
    ) {
        val next = operationActive && objectDetectionEnabled && (limiterEnabled || profilingEnabled)
        if (requested == next) return
        requested = next
        if (next) {
            try {
                val current = monitor ?: createMonitor().also { monitor = it }
                if (!current.start()) {
                    requested = false
                    stopMonitor()
                }
            } catch (_: RuntimeException) {
                requested = false
                stopMonitor()
            }
        } else {
            stopMonitor()
        }
    }

    @Synchronized
    fun score(): Double? = if (requested) monitor?.score() else null

    private fun stopMonitor() {
        try {
            monitor?.stop()
        } catch (_: RuntimeException) {}
    }
}
