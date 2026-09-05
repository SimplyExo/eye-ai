package com.algorithmic_alliance.eyeaiapp.inference

/** Runtime-owned optional sensor lifecycle. A sensor failure must not prevent image analysis. */
class PhoneMotionLifecycle(private val createMonitor: () -> PhoneMotionMonitor) {
    private var monitor: PhoneMotionMonitor? = null
    private var requested = false

    @Synchronized
    fun update(operationActive: Boolean, objectDetectionEnabled: Boolean) {
        val next = operationActive && objectDetectionEnabled
        if (requested == next) return
        requested = next
        if (next) {
            try {
                val current = monitor ?: createMonitor().also { monitor = it }
                current.start()
            } catch (_: RuntimeException) {
                // Registration/absence is optional. A subsequent enable/start retries it.
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
        } catch (_: RuntimeException) {
            // Monitor invalidates its callback generation before unregistering.
        }
    }
}
