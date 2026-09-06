package com.algorithmic_alliance.eyeaiapp.audio

import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import org.junit.Assert.*
import org.junit.Test
import java.io.Closeable
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class SpatialAudioLifecycleTest {
    private class Gate {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        fun block() {
            entered.countDown()
            check(release.await(5, TimeUnit.SECONDS)) { "barrier was not released" }
        }
        fun awaitEntry() = assertTrue(entered.await(3, TimeUnit.SECONDS))
    }

    private class Backend : SpatialAudioSessionBackend {
        private val lock = Any()
        private var next = 0uL
        val active = ConcurrentHashMap.newKeySet<ULong>()
        val engines = ConcurrentHashMap.newKeySet<ULong>()
        val created = CopyOnWriteArrayList<ULong>()
        val sent = CopyOnWriteArrayList<ULong>()
        val destroyed = CopyOnWriteArrayList<ULong>()
        val destroyThreads = CopyOnWriteArrayList<Thread>()
        var beforeCreate: (ULong) -> Unit = {}
        var beforeSend: (ULong) -> Unit = {}
        var beforeDestroy: (ULong) -> Unit = {}
        private val ready = ConcurrentHashMap<ULong, CountDownLatch>()

        override fun begin(): ULong = synchronized(lock) {
            active.clear()
            (++next).also { active.add(it) }
        }
        override fun invalidate(session: ULong) { synchronized(lock) { active.remove(session) } }
        override fun create(session: ULong) {
            created.add(session)
            beforeCreate(session)
            synchronized(lock) { if (session in active) engines.add(session) }
        }
        fun send(session: ULong) {
            beforeSend(session)
            synchronized(lock) {
                if (session in active) {
                    // Native missing-engine recovery is verified with real session code in Rust.
                    engines.add(session)
                    sent.add(session)
                    ready.computeIfAbsent(session) { CountDownLatch(1) }.countDown()
                }
            }
        }
        override fun destroy(session: ULong) {
            destroyThreads.add(Thread.currentThread())
            beforeDestroy(session)
            synchronized(lock) {
                active.remove(session)
                engines.remove(session)
                destroyed.add(session)
            }
        }
        fun awaitSend(session: ULong) = assertTrue(
            ready.computeIfAbsent(session) { CountDownLatch(1) }.await(3, TimeUnit.SECONDS),
        )
    }

    private class Fixture(val queuedWorker: Gate? = null) : Closeable {
        val backend = Backend()
        val errors = CopyOnWriteArrayList<Throwable>()
        val workers = CopyOnWriteArrayList<java.util.concurrent.ExecutorService>()
        val gates = mutableListOf<Gate>()
        val lifecycle = SpatialAudioLifecycle(
            backend,
            updates = { id -> while (isActive) { backend.send(id); delay(5) } },
            onError = { errors.add(it) },
            executorFactory = {
                Executors.newSingleThreadExecutor().also {
                    workers.add(it)
                    queuedWorker?.let { gate -> it.execute { gate.block() } }
                }
            },
        )
        fun gate() = Gate().also { gates.add(it) }
        fun awaitWorkers() {
            workers.forEach { assertTrue("session worker leaked", it.awaitTermination(5, TimeUnit.SECONDS)) }
            assertTrue(backend.engines.isEmpty())
            assertTrue(backend.active.isEmpty())
        }
        override fun close() {
            queuedWorker?.release?.countDown()
            gates.forEach { it.release.countDown() }
            lifecycle.stop()
            awaitWorkers()
        }
    }

    @Test fun blockedCreateAcrossStopCannotSurviveAndAlwaysFinalizes() {
        Fixture().use { f ->
            val gate = f.gate()
            f.backend.beforeCreate = { gate.block() }
            f.lifecycle.start()
            gate.awaitEntry()
            f.lifecycle.stop()
            assertNull(f.lifecycle.currentSessionId())
            assertTrue(f.backend.active.isEmpty())
            assertEquals(1L, gate.release.count) // stop returned before native completion
            gate.release.countDown()
            f.awaitWorkers()
            assertEquals(listOf(1uL), f.backend.destroyed)
            assertTrue(f.backend.sent.isEmpty())
            assertTrue(f.errors.isEmpty())
        }
    }

    @Test fun newSessionStartsBeforeOldCreateReturnsAndOldCleanupCannotDestroyIt() {
        Fixture().use { f ->
            val gate = f.gate()
            f.backend.beforeCreate = { if (it == 1uL) gate.block() }
            f.lifecycle.start()
            gate.awaitEntry()
            f.lifecycle.stop()
            f.lifecycle.start()
            f.backend.awaitSend(2uL)
            gate.release.countDown()
            assertTrue(f.workers[0].awaitTermination(3, TimeUnit.SECONDS))
            assertEquals(2uL, f.lifecycle.currentSessionId())
            assertEquals(setOf(2uL), f.backend.engines)
            assertFalse(f.backend.sent.contains(1uL))
            assertTrue(f.errors.isEmpty())
        }
    }

    @Test fun blockedOldSendCannotReachNewEngineOrContinueSending() {
        Fixture().use { f ->
            val gate = f.gate()
            val oldSends = AtomicInteger()
            f.backend.beforeSend = { if (it == 1uL) { oldSends.incrementAndGet(); gate.block() } }
            f.lifecycle.start()
            gate.awaitEntry()
            f.lifecycle.stop()
            f.lifecycle.start()
            f.backend.awaitSend(2uL)
            gate.release.countDown()
            assertTrue(f.workers[0].awaitTermination(3, TimeUnit.SECONDS))
            assertEquals(1, oldSends.get())
            assertEquals(setOf(2uL), f.backend.engines)
            assertFalse(f.backend.sent.contains(1uL))
            assertTrue(f.errors.isEmpty())
        }
    }

    @Test fun oldFinalizerCanBlockWhileNewSessionCreatesAndSends() {
        Fixture().use { f ->
            val gate = f.gate()
            f.backend.beforeDestroy = { if (it == 1uL) gate.block() }
            f.lifecycle.start()
            f.backend.awaitSend(1uL)
            f.lifecycle.stop()
            gate.awaitEntry()
            f.lifecycle.start()
            f.backend.awaitSend(2uL)
            gate.release.countDown()
            assertTrue(f.workers[0].awaitTermination(3, TimeUnit.SECONDS))
            assertEquals(setOf(2uL), f.backend.engines)
            assertEquals(2uL, f.lifecycle.currentSessionId())
            assertTrue(f.errors.isEmpty())
        }
    }

    @Test fun cancellationBeforeBodyStartsStillDestroysOnWorkerAndClosesExecutor() {
        val gate = Gate()
        Fixture(gate).use { f ->
            f.lifecycle.start()
            gate.awaitEntry()
            val caller = Thread.currentThread()
            f.lifecycle.stop()
            assertTrue(f.backend.created.isEmpty())
            assertTrue(f.backend.destroyed.isEmpty())
            gate.release.countDown()
            f.awaitWorkers()
            assertEquals(listOf(1uL), f.backend.destroyed)
            assertFalse(f.backend.destroyThreads.contains(caller))
            assertTrue(f.errors.isEmpty())
        }
    }

    @Test fun createAndSendExceptionsFinalizeAndAllowANewStart() {
        for (failCreate in listOf(true, false)) {
            Fixture().use { f ->
                val fail: (ULong) -> Unit = { if (it == 1uL) error("injected native failure") }
                if (failCreate) f.backend.beforeCreate = fail else f.backend.beforeSend = fail
                f.lifecycle.start()
                assertTrue(f.workers[0].awaitTermination(3, TimeUnit.SECONDS))
                assertNull(f.lifecycle.currentSessionId())
                assertEquals(listOf(1uL), f.backend.destroyed)
                assertEquals(1, f.errors.size)
                assertTrue(f.backend.engines.isEmpty())
                f.lifecycle.start()
                f.backend.awaitSend(2uL)
                assertEquals(setOf(2uL), f.backend.engines)
            }
        }
    }

    @Test fun repeatedStartStopAndRapidCyclesHaveOneFinalizerPerUniqueSession() {
        Fixture().use { f ->
            repeat(50) {
                f.lifecycle.start()
                val id = f.lifecycle.currentSessionId()
                f.lifecycle.start()
                assertEquals(id, f.lifecycle.currentSessionId())
                f.lifecycle.stop()
                f.lifecycle.stop()
            }
            f.awaitWorkers()
            assertEquals(50, f.workers.size)
            assertEquals(50, f.backend.destroyed.toSet().size)
            assertEquals(50, f.backend.destroyed.size)
            assertTrue(f.errors.isEmpty())
        }
    }

    @Test fun missingEngineCanRecoverWithinTheSameActiveSession() {
        Fixture().use { f ->
            f.lifecycle.start()
            f.backend.awaitSend(1uL)
            f.backend.engines.remove(1uL)
            f.backend.send(1uL)
            assertEquals(1uL, f.lifecycle.currentSessionId())
            assertEquals(setOf(1uL), f.backend.engines)
            f.lifecycle.stop()
            f.awaitWorkers()
            assertTrue(f.errors.isEmpty())
        }
    }
}
