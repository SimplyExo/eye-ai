package com.algorithmic_alliance.eyeaiapp.inference.throttling.motion

import com.algorithmic_alliance.eyeaiapp.inference.throttling.MonotonicClock
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PhoneMotionLifecycleTest {
	@Test
	fun runtimeLimiterAndProfilingJointlyOwnSensorLifecycle() {
		val source = Source()
		var now = 0L
		val lifecycle = PhoneMotionLifecycle {
			PhoneMotionMonitor(source, clock = MonotonicClock { now })
		}

		lifecycle.update(false, true, limiterEnabled = true, profilingEnabled = false)
		assertEquals(0, source.registrations)
		lifecycle.update(true, true, limiterEnabled = true, profilingEnabled = false)
		lifecycle.update(true, true, limiterEnabled = true, profilingEnabled = false)
		assertEquals(1, source.registrations)
		source.callbacks!!.onGyroscopeSample(2f, 0f, 0f, 0L)
		assertEquals(1.0, lifecycle.score()!!, 0.0)

		lifecycle.update(true, true, limiterEnabled = false, profilingEnabled = false)
		assertNull(lifecycle.score())
		assertEquals(1, source.unregistrations)

		lifecycle.update(true, true, limiterEnabled = false, profilingEnabled = true)
		assertEquals(2, source.registrations)
		now = 600_000_000L
		assertNull(lifecycle.score())
		lifecycle.update(false, false, limiterEnabled = false, profilingEnabled = false)
		assertEquals(2, source.unregistrations)
	}

	@Test
	fun absentFailedAndThrowingSourcesRemainOptional() {
		for (kind in 0..2) {
			val source = Source().apply {
				hasGyroscope = kind != 0
				registerResult = kind != 1
				throwOnRegister = kind == 2
			}
			val lifecycle = PhoneMotionLifecycle {
				PhoneMotionMonitor(source, clock = MonotonicClock { 0L })
			}
			lifecycle.update(true, true, limiterEnabled = true, profilingEnabled = false)
			assertNull(lifecycle.score())
			lifecycle.update(false, false, limiterEnabled = false, profilingEnabled = false)
			assertNull(lifecycle.score())
		}
	}

	@Test
	fun failedRegistrationCanBeRetriedWithoutTogglingRuntimeState() {
		val source = Source().apply { registerResult = false }
		val lifecycle = PhoneMotionLifecycle {
			PhoneMotionMonitor(source, clock = MonotonicClock { 0L })
		}

		lifecycle.update(true, true, limiterEnabled = true, profilingEnabled = false)
		assertEquals(1, source.registrations)
		source.registerResult = true
		lifecycle.update(true, true, limiterEnabled = true, profilingEnabled = false)
		assertEquals(2, source.registrations)
	}

	private class Source : PhoneMotionSensorSource {
		override var hasGyroscope = true
		override val hasLinearAcceleration = false
		var registerResult = true
		var throwOnRegister = false
		var registrations = 0
		var unregistrations = 0
		var callbacks: PhoneMotionSensorCallbacks? = null

		override fun register(callbacks: PhoneMotionSensorCallbacks): Boolean {
			registrations++
			this.callbacks = callbacks
			if (throwOnRegister) error("Registration failed")
			return registerResult
		}

		override fun unregister(callbacks: PhoneMotionSensorCallbacks) {
			unregistrations++
			this.callbacks = null
		}
	}
}
