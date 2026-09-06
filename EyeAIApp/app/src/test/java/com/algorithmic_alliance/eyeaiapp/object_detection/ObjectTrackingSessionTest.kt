package com.algorithmic_alliance.eyeaiapp.object_detection

import com.algorithmic_alliance.eyeaiapp.camera.TrackingEpoch
import org.junit.Assert.*
import org.junit.Test
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicReference

class ObjectTrackingSessionTest {
    private val a = TrackingEpoch(1, 1, 1, 1)
    private val boundaries = listOf(
        a.copy(run = 3), // stop/start
        a.copy(source = 2),
        a.copy(content = 2), // rotation, geometry, long arrival gap
        a.copy(objectDetection = 3), // disable/enable
    )

    @Test fun lateMutationFinishesBeforeResetAndNewEpochStartsTentative() {
        for (b in boundaries) {
            val session = ObjectTrackingSession()
            val current = AtomicReference(a)
            val events = Collections.synchronizedList(mutableListOf<String>())
            var evidence = 0
            fun detect(epoch: TrackingEpoch, block: () -> Unit = {}): Boolean? = session.run(
                epoch, ready = { true }, admit = { current.get() == epoch },
                reset = { events.add("reset:$epoch"); evidence = 0 },
            ) {
                block()
                events.add("mutate:$epoch")
                ++evidence >= 3
            }
            assertEquals(false, detect(a))
            assertEquals(false, detect(a))
            assertEquals(true, detect(a)) // A has CONFIRMED evidence.
            events.clear()
            val entered = CountDownLatch(1)
            val release = CountDownLatch(1)
            val newRequested = CountDownLatch(1)
            val executor = Executors.newFixedThreadPool(2)
            try {
                val old = executor.submit<Boolean?> {
                    detect(a) {
                        entered.countDown()
                        check(release.await(5, TimeUnit.SECONDS))
                    }
                }
                assertTrue(entered.await(3, TimeUnit.SECONDS))
                current.set(b) // Lifecycle boundary does not acquire the model lock.
                val fresh = executor.submit<Boolean?> {
                    newRequested.countDown()
                    detect(b)
                }
                assertTrue(newRequested.await(3, TimeUnit.SECONDS))
                assertThrows(TimeoutException::class.java) { fresh.get(100, TimeUnit.MILLISECONDS) }
                assertTrue(events.isEmpty()) // No reset ahead of old mutation.
                release.countDown()
                assertEquals(true, old.get(3, TimeUnit.SECONDS))
                assertEquals(false, fresh.get(3, TimeUnit.SECONDS))
                assertEquals(listOf("mutate:$a", "reset:$b", "mutate:$b"), events)
                // Even a delayed old request behind B cannot reset backwards.
                assertNull(detect(a))
                assertEquals(false, detect(b))
                assertEquals(true, detect(b))
            } finally {
                release.countDown()
                executor.shutdownNow()
                assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS))
            }
        }
    }

    @Test fun skippedAndNotReadyRequestsDoNotResetAndSameEpochRetainsEvidence() {
        val session = ObjectTrackingSession()
        var resets = 0
        var updates = 0
        var admissions = 0
        fun run(epoch: TrackingEpoch, ready: Boolean = true, allowed: Boolean = true) = session.run(
            epoch, { ready }, { admissions++; allowed }, { resets++ }, { ++updates },
        )
        assertEquals(1, run(a))
        for (b in boundaries) {
            assertNull(run(b, ready = false))
            assertNull(run(b, allowed = false))
        }
        assertEquals(1 + boundaries.size, admissions)
        // Cadence/mode/lost-found only changes arrival/admission, never the epoch.
        repeat(50) {
            assertNull(run(a, allowed = false))
            run(a)
        }
        assertEquals(1, resets)
        assertEquals(51, updates)
    }

    @Test fun resetFailureIsRetriedAndModelReplacementInvalidatesEpoch() {
        val session = ObjectTrackingSession()
        var failReset = true
        var resets = 0
        var operations = 0
        fun run() = session.run(a, { true }, { true }, {
            resets++
            if (failReset) error("reset failed")
        }) { ++operations }
        assertThrows(IllegalStateException::class.java) { run() }
        assertEquals(0, operations)
        failReset = false
        assertEquals(1, run())
        assertEquals(2, resets)
        assertEquals(2, run())
        assertEquals(2, resets)
        session.withModelLock { session.modelReplaced() }
        assertEquals(3, run())
        assertEquals(3, resets)
    }

    @Test fun inferenceFailureCannotLetAnOldEpochBypassTheNextReset() {
        val session = ObjectTrackingSession()
        var evidence = 0
        var resets = 0
        assertThrows(IllegalStateException::class.java) {
            session.run(a, { true }, { true }, { resets++; evidence = 0 }) {
                evidence = 3
                error("failure after tracker mutation")
            }
        }
        assertEquals(3, evidence)
        assertEquals(1, session.run(boundaries.first(), { true }, { true }, {
            resets++; evidence = 0
        }) { ++evidence })
        assertEquals(2, resets)
    }
}
