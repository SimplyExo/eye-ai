package com.algorithmic_alliance.eyeaiapp.runtime

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RuntimeLifecycleGateTest {
    @Test
    fun `runtime start and stop are idempotent`() {
        val gate = RuntimeLifecycleGate()

        assertTrue(gate.start())
        assertFalse(gate.start())
        assertTrue(gate.isActive)

        assertTrue(gate.stop())
        assertFalse(gate.stop())
        assertFalse(gate.isActive)
    }

    @Test
    fun `runtime can be started again after an explicit stop`() {
        val gate = RuntimeLifecycleGate()

        assertTrue(gate.start())
        assertTrue(gate.stop())
        assertTrue(gate.start())
        assertTrue(gate.isActive)
    }
}
