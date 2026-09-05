package com.algorithmic_alliance.eyeaiapp.inference

import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import java.util.concurrent.Callable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class InferenceSchedulerTest {
	@Test
	fun `first inference is immediate and an acquisition is consumed`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		assertTrue(scheduler.tryAcquireInference())
		assertFalse(scheduler.tryAcquireInference())
	}

	@Test
	fun `quiet mode observes its finite safety floor`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.tryAcquireInference()
		clock.advance(999.milliseconds)
		assertFalse(scheduler.tryAcquireInference())
		clock.advance(1.milliseconds)
		assertTrue(scheduler.tryAcquireInference())
	}

	@Test
	fun `active mode uses a shorter interval than quiet`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updateVisualChange(0.5)
		assertEquals(InferenceMode.ACTIVE, scheduler.mode)
		assertTrue(scheduler.tryAcquireInference())

		clock.advance(199.milliseconds)
		assertFalse(scheduler.tryAcquireInference())
		clock.advance(1.milliseconds)
		assertTrue(scheduler.tryAcquireInference())
	}

	@Test
	fun `burst mode uses a shorter interval than active`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updateVisualChange(0.9)
		assertEquals(InferenceMode.BURST, scheduler.mode)
		assertTrue(scheduler.tryAcquireInference())

		clock.advance(49.milliseconds)
		assertFalse(scheduler.tryAcquireInference())
		clock.advance(1.milliseconds)
		assertTrue(scheduler.tryAcquireInference())
	}

	@Test
	fun `strong scene change immediately escalates quiet cadence`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.tryAcquireInference()
		clock.advance(100.milliseconds)
		scheduler.updateVisualChange(0.9)

		assertEquals(InferenceMode.BURST, scheduler.mode)
		assertTrue(scheduler.tryAcquireInference())
	}

	@Test
	fun `burst remains active for its hold time even after the signal becomes stale`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updateVisualChange(0.9)
		clock.advance(499.milliseconds)
		assertEquals(InferenceMode.BURST, scheduler.mode)

		clock.advance(2.milliseconds)
		assertEquals(InferenceMode.QUIET, scheduler.mode)
	}

	@Test
	fun `a new strong scene signal extends burst hold`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updateVisualChange(0.9)
		clock.advance(400.milliseconds)
		scheduler.updateVisualChange(0.9)
		clock.advance(499.milliseconds)

		assertEquals(InferenceMode.BURST, scheduler.mode)
		clock.advance(2.milliseconds)
		assertEquals(InferenceMode.QUIET, scheduler.mode)
	}

	@Test
	fun `active hysteresis prevents flutter around entry threshold`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updateVisualChange(0.41)
		assertEquals(InferenceMode.ACTIVE, scheduler.mode)
		clock.advance(10.milliseconds)
		scheduler.updateVisualChange(0.39)
		assertEquals(InferenceMode.ACTIVE, scheduler.mode)
		clock.advance(10.milliseconds)
		scheduler.updateVisualChange(0.41)
		assertEquals(InferenceMode.ACTIVE, scheduler.mode)
	}

	@Test
	fun `active mode returns to quiet only after low activity hold`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updateVisualChange(0.5)
		clock.advance(10.milliseconds)
		scheduler.updateVisualChange(0.0)
		clock.advance(299.milliseconds)
		assertEquals(InferenceMode.ACTIVE, scheduler.mode)
		clock.advance(1.milliseconds)
		assertEquals(InferenceMode.QUIET, scheduler.mode)
	}

	@Test
	fun `null phone motion does not create activity`() {
		val scheduler = scheduler(FakeMonotonicClock())

		scheduler.updateActivity(visualChangeScore = 0.0, phoneMotionScore = null)

		assertEquals(InferenceMode.QUIET, scheduler.mode)
	}

	@Test
	fun `stale phone motion eventually stops keeping scheduler active`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updateActivity(visualChangeScore = 0.0, phoneMotionScore = 0.9)
		assertEquals(InferenceMode.ACTIVE, scheduler.mode)
		clock.advance(201.milliseconds)
		assertEquals(InferenceMode.ACTIVE, scheduler.mode)
		clock.advance(299.milliseconds)
		assertEquals(InferenceMode.QUIET, scheduler.mode)
	}

	@Test
	fun `phone motion alone can reach active but never burst`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updateActivity(visualChangeScore = 0.0, phoneMotionScore = 0.9)

		assertEquals(InferenceMode.ACTIVE, scheduler.mode)
	}

	@Test
	fun `a long frame pause does not accumulate inference slots`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.tryAcquireInference()
		clock.advance(10_000.milliseconds)

		assertTrue(scheduler.tryAcquireInference())
		assertFalse(scheduler.tryAcquireInference())
		clock.advance(999.milliseconds)
		assertFalse(scheduler.tryAcquireInference())
		clock.advance(1.milliseconds)
		assertTrue(scheduler.tryAcquireInference())
	}

	@Test
	fun `reset clears mode and cadence`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updateVisualChange(0.5)
		scheduler.tryAcquireInference()
		clock.advance(100.milliseconds)
		scheduler.reset()

		assertEquals(InferenceMode.QUIET, scheduler.mode)
		assertTrue(scheduler.tryAcquireInference())
	}

	@Test
	fun `injected maximum rate caps every mode interval`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock, maxObjectDetectionRateHz = 2.0)

		scheduler.updateVisualChange(0.5)
		assertEquals(InferenceMode.ACTIVE, scheduler.mode)
		scheduler.tryAcquireInference()
		clock.advance(400.milliseconds)
		scheduler.updateVisualChange(0.5)
		clock.advance(99.milliseconds)
		assertFalse(scheduler.tryAcquireInference())
		clock.advance(1.milliseconds)
		assertTrue(scheduler.tryAcquireInference())
	}

	@Test
	fun `quiet to burst cannot bypass the hard cap`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock, maxObjectDetectionRateHz = 2.0)

		assertTrue(scheduler.tryAcquireInference())
		clock.advance(100.milliseconds)
		scheduler.updateVisualChange(0.9)

		assertEquals(InferenceMode.BURST, scheduler.mode)
		assertFalse(scheduler.tryAcquireInference())
		clock.advance(400.milliseconds)
		assertTrue(scheduler.tryAcquireInference())
	}

	@Test
	fun `active to burst can skip active interval but not hard cap`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock, maxObjectDetectionRateHz = 2.0)

		scheduler.updateVisualChange(0.5)
		assertTrue(scheduler.tryAcquireInference())
		clock.advance(100.milliseconds)
		scheduler.updateVisualChange(0.9)

		assertEquals(InferenceMode.BURST, scheduler.mode)
		assertFalse(scheduler.tryAcquireInference())
		clock.advance(400.milliseconds)
		assertTrue(scheduler.tryAcquireInference())
	}

	@Test
	fun `changing the cap during an active burst changes only eligibility`() {
		val clock = FakeMonotonicClock()
		val cappedScheduler = scheduler(clock, maxObjectDetectionRateHz = 2.0)

		assertTrue(cappedScheduler.tryAcquireInference())
		clock.advance(100.milliseconds)
		cappedScheduler.updateVisualChange(0.9)
		assertFalse(cappedScheduler.tryAcquireInference())

		cappedScheduler.updateConfig(config(maxObjectDetectionRateHz = null), atNanos = clock.now)
		assertEquals(InferenceMode.BURST, cappedScheduler.mode)
		assertTrue(cappedScheduler.tryAcquireInference())
		assertFalse(cappedScheduler.tryAcquireInference())
	}

	@Test
	fun `mode and policy refresh do not double consume a slot`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock, maxObjectDetectionRateHz = 2.0)

		assertTrue(scheduler.tryAcquireInference())
		clock.advance(100.milliseconds)
		scheduler.updateVisualChange(0.9)
		scheduler.updateConfig(config(maxObjectDetectionRateHz = 2.0), atNanos = clock.now)
		scheduler.updateVisualChange(0.9, atNanos = clock.now)

		assertFalse(scheduler.tryAcquireInference())
		clock.advance(400.milliseconds)
		assertTrue(scheduler.tryAcquireInference())
		assertFalse(scheduler.tryAcquireInference())
	}

	@Test
	fun `fake clock is monotone and backwards operation time is rejected`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)
		clock.advance(10.milliseconds)
		scheduler.snapshot()

		try {
			scheduler.snapshot(atNanos = 9.milliseconds.inWholeNanoseconds)
			fail("expected monotonic time validation to reject a backwards timestamp")
		} catch (_: IllegalArgumentException) {
			// Expected.
		}
	}

	@Test
	fun `old visual sample is rejected after reset`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updateVisualChange(0.9, atNanos = 0L)
		scheduler.reset(atNanos = 100.milliseconds.inWholeNanoseconds)
		scheduler.updateVisualChange(0.9, atNanos = 99.milliseconds.inWholeNanoseconds)

		assertEquals(InferenceMode.QUIET, scheduler.snapshot(atNanos = 100.milliseconds.inWholeNanoseconds).mode)
		scheduler.updateVisualChange(0.9, atNanos = 101.milliseconds.inWholeNanoseconds)
		assertEquals(InferenceMode.BURST, scheduler.snapshot(atNanos = 101.milliseconds.inWholeNanoseconds).mode)
	}

	@Test
	fun `old phone sample cannot revive a newer explicit absence`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updatePhoneMotion(0.9, atNanos = 0L)
		scheduler.updatePhoneMotion(null, atNanos = 100.milliseconds.inWholeNanoseconds)
		scheduler.updatePhoneMotion(0.9, atNanos = 99.milliseconds.inWholeNanoseconds)

		assertEquals(InferenceMode.QUIET, scheduler.snapshot(atNanos = 400.milliseconds.inWholeNanoseconds).mode)
		scheduler.updatePhoneMotion(0.9, atNanos = 401.milliseconds.inWholeNanoseconds)
		assertEquals(InferenceMode.ACTIVE, scheduler.snapshot(atNanos = 401.milliseconds.inWholeNanoseconds).mode)
	}

	@Test
	fun `newer null clears a valid phone hint after the quiet hold`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updatePhoneMotion(0.9, atNanos = 0L)
		scheduler.updatePhoneMotion(null, atNanos = 100.milliseconds.inWholeNanoseconds)

		assertEquals(
			InferenceMode.QUIET,
			scheduler.snapshot(atNanos = 400.milliseconds.inWholeNanoseconds).mode,
		)
	}

	@Test
	fun `repeated absence and out of order samples remain deterministic`() {
		val clock = FakeMonotonicClock()
		val scheduler = scheduler(clock)

		scheduler.updatePhoneMotion(0.9, atNanos = 10.milliseconds.inWholeNanoseconds)
		scheduler.updatePhoneMotion(null, atNanos = 20.milliseconds.inWholeNanoseconds)
		scheduler.updatePhoneMotion(null, atNanos = 19.milliseconds.inWholeNanoseconds)
		scheduler.updatePhoneMotion(0.9, atNanos = 19.milliseconds.inWholeNanoseconds)

		assertEquals(
			InferenceMode.QUIET,
			scheduler.snapshot(atNanos = 320.milliseconds.inWholeNanoseconds).mode,
		)
		scheduler.updatePhoneMotion(0.9, atNanos = 321.milliseconds.inWholeNanoseconds)
		assertEquals(
			InferenceMode.ACTIVE,
			scheduler.snapshot(atNanos = 321.milliseconds.inWholeNanoseconds).mode,
		)
	}

	@Test
	fun `interval ordering is validated without selecting product values`() {
		try {
			config(quietInterval = 100.milliseconds, activeInterval = 200.milliseconds)
			fail("expected quiet interval ordering validation")
		} catch (_: IllegalArgumentException) {
			// Expected.
		}

		try {
			config(activeInterval = 200.milliseconds, burstInterval = 300.milliseconds)
			fail("expected burst interval ordering validation")
		} catch (_: IllegalArgumentException) {
			// Expected.
		}
	}

	@Test
	fun `concurrent acquisitions consume exactly one slot`() {
		val scheduler = scheduler(FakeMonotonicClock())
		val start = CountDownLatch(1)
		val executor = Executors.newFixedThreadPool(2)

		try {
			val futures = listOf(
				executor.submit(Callable {
					start.await()
					scheduler.tryAcquireInference()
				}),
				executor.submit(Callable {
					start.await()
					scheduler.tryAcquireInference()
				}),
			)
			start.countDown()

			val results = futures.map { it.get(1, TimeUnit.SECONDS) }
			assertEquals(1, results.count { it })
		} finally {
			executor.shutdownNow()
			executor.awaitTermination(1, TimeUnit.SECONDS)
		}
	}

	@Test
	fun `signal update and reset can run parallel to acquire without time races`() {
		val scheduler = scheduler(FakeMonotonicClock())
		val start = CountDownLatch(1)
		val failures = AtomicReference<Throwable?>(null)
		val executor = Executors.newFixedThreadPool(3)

		try {
			val jobs = listOf(
				Runnable {
					try {
						start.await()
						scheduler.updateVisualChange(0.5)
					} catch (error: Throwable) {
						failures.compareAndSet(null, error)
					}
				},
				Runnable {
					try {
						start.await()
						scheduler.reset()
					} catch (error: Throwable) {
						failures.compareAndSet(null, error)
					}
				},
				Runnable {
					try {
						start.await()
						scheduler.tryAcquireInference()
					} catch (error: Throwable) {
						failures.compareAndSet(null, error)
					}
				},
			)
			jobs.forEach(executor::submit)
			start.countDown()
			executor.shutdown()
			assertTrue(executor.awaitTermination(1, TimeUnit.SECONDS))
			assertNull(failures.get())
		} finally {
			if (!executor.isTerminated) executor.shutdownNow()
		}
	}

	@Test
	fun `normal clock read is serialized before another operation can advance time`() {
		val backgroundStarted = CountDownLatch(1)
		val backgroundFinished = CountDownLatch(1)
		val backgroundFailure = AtomicReference<Throwable?>(null)
		var clockReads = 0
		lateinit var scheduler: InferenceScheduler
		val clock = object : MonotonicClock {
			override fun nowNanos(): Long {
				if (clockReads++ == 0) return 0L
				backgroundStarted.countDown()
				Thread {
					try {
						scheduler.snapshot(atNanos = 100L)
					} catch (error: Throwable) {
						backgroundFailure.set(error)
					} finally {
						backgroundFinished.countDown()
					}
				}.start()
				backgroundFinished.await(200, TimeUnit.MILLISECONDS)
				return 99L
			}
		}
		scheduler = InferenceScheduler(config(), clock)

		assertTrue(scheduler.tryAcquireInference())
		assertTrue(backgroundStarted.await(1, TimeUnit.SECONDS))
		assertTrue(backgroundFinished.await(1, TimeUnit.SECONDS))
		assertNull(backgroundFailure.get())
	}

	private fun scheduler(
		clock: FakeMonotonicClock,
		maxObjectDetectionRateHz: Double? = null,
	): InferenceScheduler = InferenceScheduler(config(maxObjectDetectionRateHz), clock)

	private fun config(
		maxObjectDetectionRateHz: Double? = null,
		quietInterval: Duration = 1_000.milliseconds,
		activeInterval: Duration = 200.milliseconds,
		burstInterval: Duration = 50.milliseconds,
	): InferenceSchedulerConfig = InferenceSchedulerConfig(
		quietInterval = quietInterval,
		activeInterval = activeInterval,
		burstInterval = burstInterval,
		quietHoldTime = 300.milliseconds,
		burstHoldTime = 500.milliseconds,
		signalTimeout = 200.milliseconds,
		activeVisualEntryThreshold = 0.4,
		activeVisualExitThreshold = 0.25,
		quietVisualThreshold = 0.1,
		activeMotionEntryThreshold = 0.5,
		activeMotionExitThreshold = 0.3,
		quietMotionThreshold = 0.1,
		burstVisualEntryThreshold = 0.8,
		maxObjectDetectionRateHz = maxObjectDetectionRateHz,
	)

	private class FakeMonotonicClock(
		var now: Long = 0L,
	) : MonotonicClock {
		override fun nowNanos(): Long = now

		fun advance(duration: Duration) {
			now += duration.inWholeNanoseconds
		}
	}
}
