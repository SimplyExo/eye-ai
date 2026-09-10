package com.algorithmic_alliance.eyeaiapp.camera

import android.graphics.Bitmap
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

class LatestFramePipelineInstrumentationTest {
    @Test fun latestSlotDropsBacklogAndNeverReinfersItsSequence() {
        PipelineFixture().use { f ->
            val gate = PipelineGate(); f.backend.objectGate = gate
            val released = CountDownLatch(3)
            f.send(0, release = { released.countDown() })
            assertTrue(gate.entered.await(3, TimeUnit.SECONDS))
            f.send(400, release = { released.countDown() })
            val newest = pipelinePixels()
            f.send(800, newest, release = { released.countDown() })
            gate.release.countDown()
            val first = f.result(); val last = f.result()
            assertEquals(1L, first.sequence)
            assertEquals(3L, last.sequence)
            assertEquals(2, f.backend.trackerSteps.get())
            assertNotNull(f.backend.inferred.poll(3, TimeUnit.SECONDS))
            assertSame(newest, f.backend.inferred.poll(3, TimeUnit.SECONDS))
            f.time.set(2_000_000_000)
            assertNull(f.updates.poll(100, TimeUnit.MILLISECONDS))
            f.analyzer.stop()
            assertTrue(released.await(3, TimeUnit.SECONDS))
        }
    }

    @Test fun blockedObjectDetectionDoesNotThrottleDepthAndDepthUsesNewestFrame() {
        PipelineFixture().use { f ->
            val objectGate = PipelineGate()
            val depthGate = PipelineGate()
            f.backend.objectGate = objectGate
            f.backend.depthGate = depthGate
            f.backend.captureDepth = true

            val first = pipelinePixels()
            f.send(0, first)
            assertTrue(objectGate.entered.await(3, TimeUnit.SECONDS))
            assertTrue(depthGate.entered.await(3, TimeUnit.SECONDS))
            assertSame(first, f.backend.depthInferred.poll(3, TimeUnit.SECONDS))

            f.send(10, pipelinePixels(true))
            val newest = Bitmap.createBitmap(16, 12, Bitmap.Config.ARGB_8888)
            f.send(20, newest)

            depthGate.release.countDown()
            assertSame(newest, f.backend.depthInferred.poll(3, TimeUnit.SECONDS))
            assertNull(f.backend.depthInferred.poll(100, TimeUnit.MILLISECONDS))

            objectGate.release.countDown()
        }
    }

    @Test fun ownershipSurvivesConcurrentDepthOcrAndStop() {
        PipelineFixture().use { f ->
            val depthGate = PipelineGate(); val ocrGate = PipelineGate()
            f.backend.depthGate = depthGate; f.backend.ocrGate = ocrGate
            val released = CountDownLatch(1)
            val count = AtomicInteger()
            f.send(0, release = { count.incrementAndGet(); released.countDown() })
            f.admitted(); f.result()
            assertTrue(depthGate.entered.await(3, TimeUnit.SECONDS))
            val ocrResult = AtomicReference<Boolean?>()
            val ocr = Thread { runBlocking { ocrResult.set(f.analyzer.runOcrAnalysis()) } }
            ocr.start()
            assertTrue(ocrGate.entered.await(3, TimeUnit.SECONDS))
            f.analyzer.stop()
            assertEquals(0, count.get())
            depthGate.release.countDown(); ocrGate.release.countDown()
            ocr.join(3_000)
            assertFalse(ocr.isAlive)
            assertEquals(false, ocrResult.get())
            assertTrue(released.await(3, TimeUnit.SECONDS))
            assertEquals(1, count.get())
            assertNull(f.store.get().depth)
        }
    }

    @Test fun rejectionAndDoubleReleaseAreDetected() {
        PipelineFixture().use { f ->
            f.analyzer.stop()
            val count = AtomicInteger()
            val frame = AnalysisFrame.fromBitmap(pipelinePixels(), onReleased = { count.incrementAndGet() })
            assertFalse(f.source.submitFrame(frame))
            assertEquals(1, count.get())
            assertFalse(frame.tryRetain())
            assertThrows(IllegalStateException::class.java) { frame.release() }
            assertEquals(1, count.get())
        }
    }

    @Test fun samplingFailureReleasesInitialReferenceWithoutReplacingLatest() {
        PipelineFixture().use { f ->
            f.send(0); f.admitted(); val previous = f.result()
            val released = AtomicInteger()
            val bitmap = pipelinePixels()
            val frame = AnalysisFrame.fromBitmap(bitmap, onReleased = { released.incrementAndGet() })
            bitmap.recycle()
            f.time.set(100_000_000)
            assertThrows(IllegalArgumentException::class.java) { f.source.submitFrame(frame) }
            assertEquals(1, released.get())
            assertFalse(frame.tryRetain())
            assertSame(previous, f.store.get().objects)
            f.send(400); f.admitted()
            assertEquals(previous.sequence + 1, f.result().sequence)
        }
    }
}
