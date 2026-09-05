package com.algorithmic_alliance.eyeaiapp.inference

import kotlin.time.Duration.Companion.milliseconds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class PhoneMotionMonitorTest {
	@Test
	fun `missing sensors do not become an artificial zero`() {
		val source = FakeSensorSource(hasGyroscope = false, hasLinearAcceleration = false)
		val monitor = PhoneMotionMonitor(source)

		assertFalse(monitor.start())
		assertNull(monitor.score(atNanos = 0L))
		assertEquals(0, source.registerCalls)
	}

	@Test
	fun `start is idempotent`() {
		val source = FakeSensorSource(hasGyroscope = true, hasLinearAcceleration = false)
		val monitor = PhoneMotionMonitor(source)

		assertTrue(monitor.start())
		assertTrue(monitor.start())

		assertEquals(1, source.registerCalls)
		assertTrue(monitor.isRunning)
	}

	@Test
	fun `stop is idempotent and callbacks after stop are ignored`() {
		val source = FakeSensorSource(hasGyroscope = true, hasLinearAcceleration = false)
		val monitor = PhoneMotionMonitor(source)

		monitor.start()
		source.emitGyroscope(1f, 0f, 0f, timestampNanos = 0L)
		assertTrue(monitor.score(atNanos = 0L)!! > 0.0)

		monitor.stop()
		monitor.stop()
		source.emitGyroscope(1f, 0f, 0f, timestampNanos = 1L)

		assertEquals(1, source.unregisterCalls)
		assertFalse(monitor.isRunning)
		assertNull(monitor.score(atNanos = 1L))
	}

	@Test
	fun `stale sensor signal becomes null`() {
		val source = FakeSensorSource(hasGyroscope = true, hasLinearAcceleration = false)
		val monitor = PhoneMotionMonitor(
			sensorSource = source,
			config = PhoneMotionMonitorConfig(staleTimeout = 100.milliseconds, smoothingFactor = 1.0),
		)

		monitor.start()
		source.emitGyroscope(1f, 0f, 0f, timestampNanos = 0L)

		assertTrue(monitor.score(atNanos = 100.milliseconds.inWholeNanoseconds)!! > 0.0)
		assertNull(monitor.score(atNanos = 101.milliseconds.inWholeNanoseconds))
	}

	@Test
	fun `reset drops the current score and rotates the callback generation`() {
		val source = FakeSensorSource(hasGyroscope = true, hasLinearAcceleration = false)
		val monitor = PhoneMotionMonitor(source)

		monitor.start()
		val oldCallbacks = source.callbackAt(0)
		source.emitGyroscope(1f, 0f, 0f, timestampNanos = 0L)
		assertTrue(monitor.score(atNanos = 0L)!! > 0.0)

		monitor.reset()
		val newCallbacks = source.callbackAt(1)
		source.emitGyroscopeFrom(oldCallbacks, 1f, 0f, 0f, timestampNanos = 1L)

		assertNull(monitor.score(atNanos = 1L))
		assertEquals(1, source.unregisterCalls)
		assertTrue(oldCallbacks !== newCallbacks)
		assertEquals(2, source.registerCalls)
		assertTrue(monitor.isRunning)
	}

	@Test
	fun `restart begins with no stale score`() {
		val source = FakeSensorSource(hasGyroscope = true, hasLinearAcceleration = false)
		val monitor = PhoneMotionMonitor(source)

		monitor.start()
		source.emitGyroscope(1f, 0f, 0f, timestampNanos = 0L)
		assertTrue(monitor.score(atNanos = 0L)!! > 0.0)
		monitor.stop()
		assertTrue(monitor.start())

		assertNull(monitor.score(atNanos = 1L))
		assertEquals(2, source.registerCalls)
	}

	@Test
	fun `callback from a stopped session cannot affect a restarted session`() {
		val source = FakeSensorSource(hasGyroscope = true, hasLinearAcceleration = false)
		val monitor = PhoneMotionMonitor(source)

		assertTrue(monitor.start())
		val oldCallbacks = source.callbackAt(0)
		monitor.stop()
		assertTrue(monitor.start())
		val newCallbacks = source.callbackAt(1)

		source.emitGyroscopeFrom(oldCallbacks, 1f, 0f, 0f, timestampNanos = 100L)
		assertNull(monitor.score(atNanos = 100L))
		source.emitGyroscopeFrom(newCallbacks, 1f, 0f, 0f, timestampNanos = 100L)
		assertTrue(monitor.score(atNanos = 100L)!! > 0.0)
	}

	@Test
	fun `start false leaves the monitor stopped`() {
		val source = FakeSensorSource(
			hasGyroscope = true,
			hasLinearAcceleration = false,
			registerResult = false,
		)
		val monitor = PhoneMotionMonitor(source)

		assertFalse(monitor.start())
		assertFalse(monitor.isRunning)
		assertNull(monitor.score(atNanos = 0L))
		assertEquals(1, source.registerCalls)
	}

	@Test
	fun `registration exception leaves partial registration cleaned up`() {
		val source = FakeSensorSource(
			hasGyroscope = true,
			hasLinearAcceleration = false,
			throwAfterRegistration = true,
		)
		val monitor = PhoneMotionMonitor(source)

		try {
			monitor.start()
			fail("expected registration failure")
		} catch (_: IllegalStateException) {
			// Expected.
		}

		assertFalse(monitor.isRunning)
		assertEquals(1, source.unregisterCalls)
		assertNull(monitor.score(atNanos = 0L))
	}

	@Test
	fun `linear acceleration can provide a valid fallback`() {
		val source = FakeSensorSource(hasGyroscope = false, hasLinearAcceleration = true)
		val monitor = PhoneMotionMonitor(
			sensorSource = source,
			config = PhoneMotionMonitorConfig(
				smoothingFactor = 1.0,
				linearAccelerationFullScaleMetersPerSecondSquared = 2.0,
			),
		)

		assertTrue(monitor.start())
		source.emitLinearAcceleration(0f, 2f, 0f, timestampNanos = 0L)

		assertEquals(1.0, monitor.score(atNanos = 0L)!!, 0.0)
	}

	private class FakeSensorSource(
		override val hasGyroscope: Boolean,
		override val hasLinearAcceleration: Boolean,
		private val registerResult: Boolean = true,
		private val throwAfterRegistration: Boolean = false,
	) : PhoneMotionSensorSource {
		var registerCalls = 0
		var unregisterCalls = 0
		private var activeCallbacks: PhoneMotionSensorCallbacks? = null
		private val savedCallbacks = mutableListOf<PhoneMotionSensorCallbacks>()

		override fun register(callbacks: PhoneMotionSensorCallbacks): Boolean {
		registerCalls++
		if (!registerResult || (!hasGyroscope && !hasLinearAcceleration)) return false
		activeCallbacks = callbacks
		savedCallbacks += callbacks
		if (throwAfterRegistration) throw IllegalStateException("fake registration failure")
		return true
	}

		override fun unregister(callbacks: PhoneMotionSensorCallbacks) {
			if (activeCallbacks === callbacks) {
				activeCallbacks = null
				unregisterCalls++
			}
		}

		fun emitGyroscope(x: Float, y: Float, z: Float, timestampNanos: Long) {
			activeCallbacks?.onGyroscopeSample(x, y, z, timestampNanos)
		}

		fun emitLinearAcceleration(x: Float, y: Float, z: Float, timestampNanos: Long) {
			activeCallbacks?.onLinearAccelerationSample(x, y, z, timestampNanos)
		}

		fun callbackAt(index: Int): PhoneMotionSensorCallbacks = savedCallbacks[index]

		fun emitGyroscopeFrom(
			callbacks: PhoneMotionSensorCallbacks,
			x: Float,
			y: Float,
			z: Float,
			timestampNanos: Long,
		) {
			callbacks.onGyroscopeSample(x, y, z, timestampNanos)
		}
	}
}
