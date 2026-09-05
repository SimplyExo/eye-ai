package com.algorithmic_alliance.eyeaiapp.inference

import org.junit.Assert.*
import org.junit.Test

class PhoneMotionLifecycleTest {
    private class Source : PhoneMotionSensorSource {
        override var hasGyroscope = true
        override val hasLinearAcceleration = false
        var result = true
        var fail = false
        var registrations = 0
        var unregistrations = 0
        var callbacks: PhoneMotionSensorCallbacks? = null
        override fun register(callbacks: PhoneMotionSensorCallbacks): Boolean {
            registrations++
            this.callbacks = callbacks
            if (fail) error("Registration failed")
            return result
        }
        override fun unregister(callbacks: PhoneMotionSensorCallbacks) {
            unregistrations++
            this.callbacks = null
        }
    }

    @Test fun runtimeAndOdJointlyOwnSensorLifecycle() {
        val source = Source()
        var now = 0L
        val owner = PhoneMotionLifecycle { PhoneMotionMonitor(source, clock = PhoneMotionClock { now }) }
        owner.update(false, true)
        assertEquals(0, source.registrations)
        owner.update(true, true); owner.update(true, true)
        assertEquals(1, source.registrations)
        val oldCallback = source.callbacks!!
        oldCallback.onGyroscopeSample(2f, 0f, 0f, 0L)
        assertEquals(1.0, owner.score()!!, 0.0)
        now = 600_000_000
        assertNull(owner.score())
        owner.update(true, false)
        assertNull(owner.score())
        owner.update(true, true)
        oldCallback.onGyroscopeSample(2f, 0f, 0f, now)
        assertNull(owner.score())
        owner.update(false, true); owner.update(false, false)
        assertEquals(2, source.unregistrations)
    }

    @Test fun absentFalseAndThrowingSourcesAllPropagateNull() {
        for (kind in 0..2) {
            val source = Source().apply {
                hasGyroscope = kind != 0
                result = kind != 1
                fail = kind == 2
            }
            val owner = PhoneMotionLifecycle { PhoneMotionMonitor(source, clock = PhoneMotionClock { 0L }) }
            owner.update(true, true)
            assertNull(owner.score())
            owner.update(false, false)
            assertNull(owner.score())
        }
    }
}
