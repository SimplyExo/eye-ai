package com.algorithmic_alliance.eyeaiapp.camera

import android.graphics.Bitmap
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class PipelineEpochInstrumentationTest {
    @Test fun stopDuringNativeCallDiscardsResultAndReleasesConsumer() {
        PipelineFixture().use { f ->
            val gate = PipelineGate(); f.backend.objectGate = gate
            val released = CountDownLatch(1)
            f.send(0, release = { released.countDown() })
            assertTrue(gate.entered.await(3, TimeUnit.SECONDS))
            f.analyzer.stop()
            assertNull(f.store.get().objects)
            assertEquals(1L, released.count)
            gate.release.countDown()
            assertTrue(released.await(3, TimeUnit.SECONDS))
            assertNull(f.store.get().objects)
            assertNull(f.updates.poll())
        }
    }

    @Test fun rapidStopStartRejectsOldResultAndOldSourceCallback() {
        PipelineFixture().use { f ->
            val gate = PipelineGate(); f.backend.objectGate = gate
            val released = CountDownLatch(1)
            f.send(0, release = { released.countDown() })
            assertTrue(gate.entered.await(3, TimeUnit.SECONDS))
            val oldSource = f.source
            val oldEpoch = f.store.get().generation.trackingEpoch
            f.analyzer.stop(); f.analyzer.start()
            f.source = f.analyzer.beginSourceSession()
            val rejected = AtomicInteger()
            assertFalse(oldSource.submitFrame(AnalysisFrame.fromBitmap(pipelinePixels(), onReleased = { rejected.incrementAndGet() })))
            assertEquals(1, rejected.get())
            f.send(400)
            gate.release.countDown()
            val result = f.result()
            assertEquals(f.store.get().generation, result.generation)
            assertNotEquals(oldEpoch, result.generation.trackingEpoch)
            assertEquals(400_000_000L, result.frameArrivalNanos)
            assertTrue(released.await(3, TimeUnit.SECONDS))
            assertNull(f.updates.poll())
        }
    }

    @Test fun disableDuringInferenceInvalidatesOnlyObjectGeneration() {
        PipelineFixture().use { f ->
            val gate = PipelineGate(); f.backend.objectGate = gate
            val released = CountDownLatch(1)
            f.send(0, release = { released.countDown() })
            assertTrue(gate.entered.await(3, TimeUnit.SECONDS))
            val before = f.store.get().generation
            val beforeEpoch = before.trackingEpoch
            f.analyzer.configureObjectDetection(false, null)
            f.send(400)
            gate.release.countDown()
            assertTrue(released.await(3, TimeUnit.SECONDS))
            assertNull(f.store.get().objects)
            assertEquals(before.source, f.store.get().generation.source)
            assertNotEquals(before.objectDetection, f.store.get().generation.objectDetection)
            assertNotEquals(beforeEpoch, f.store.get().generation.trackingEpoch)
            val disabledEpoch = f.store.get().generation.trackingEpoch
            f.analyzer.configureObjectDetection(true, null)
            assertNotEquals(disabledEpoch, f.store.get().generation.trackingEpoch)
            f.send(800)
            assertEquals(800_000_000L, f.result().frameArrivalNanos)
        }
    }

    @Test fun sourceSwitchSameGeometryInvalidatesBothModelsAndBaseline() {
        PipelineFixture().use { f ->
            val objectGate = PipelineGate(); val depthGate = PipelineGate()
            f.backend.objectGate = objectGate; f.backend.depthGate = depthGate
            val released = CountDownLatch(1)
            f.send(0, release = { released.countDown() })
            assertTrue(objectGate.entered.await(3, TimeUnit.SECONDS))
            assertTrue(depthGate.entered.await(3, TimeUnit.SECONDS))
            val oldEpoch = f.store.get().generation.trackingEpoch
            val oldSource = f.source
            f.source = f.analyzer.beginSourceSession()
            assertFalse(oldSource.isCurrent())
            assertNotEquals(oldEpoch, f.store.get().generation.trackingEpoch)
            f.send(400, pipelinePixels(true))
            assertEquals(0.0, f.analyzer.telemetrySnapshot().visualChangeScore ?: -1.0, 0.0)
            objectGate.release.countDown(); depthGate.release.countDown()
            assertTrue(released.await(3, TimeUnit.SECONDS))
            assertEquals(400_000_000L, f.result().frameArrivalNanos)
            assertNull(f.store.get().depth)
        }
    }

    @Test fun geometryRotationAndLongGapDiscardVisualEvidenceButPreserveSink() {
        PipelineFixture().use { f ->
            f.send(0); f.admitted(); f.result()
            f.send(100, pipelinePixels(true)); f.admitted(); f.result()
            val oldGeneration = f.store.get().generation
            val oldEpoch = oldGeneration.trackingEpoch
            f.send(200, rotation = 90); f.skippedBeforeAdmission()
            assertNull(f.store.get().objects)
            assertNotEquals(oldGeneration.content, f.store.get().generation.content)
            assertNotEquals(oldEpoch, f.store.get().generation.trackingEpoch)
            val rotationEpoch = f.store.get().generation.trackingEpoch
            f.send(6_000, rotation = 90); f.admitted(); f.result()
            assertTrue(f.source.isCurrent())
            assertNotEquals(rotationEpoch, f.store.get().generation.trackingEpoch)
            f.send(6_050, rotation = 90); f.skippedBeforeAdmission()
            val wider = Bitmap.createBitmap(32, 12, Bitmap.Config.ARGB_8888)
            val preResizeGeneration = f.store.get().generation
            f.send(6_100, wider, 90); f.skippedBeforeAdmission()
            assertTrue(f.source.isCurrent())
            assertNotEquals(preResizeGeneration.content, f.store.get().generation.content)
            assertNotEquals(preResizeGeneration.trackingEpoch, f.store.get().generation.trackingEpoch)
        }
    }

    @Test fun lowRateSourceDoesNotResetOnEveryFrame() {
        PipelineFixture().use { f ->
            f.send(0); f.admitted(); f.result()
            val generation = f.store.get().generation
            f.send(2_000); f.admitted(); f.result()
            assertEquals(generation, f.store.get().generation)
            assertEquals(generation.trackingEpoch, f.store.get().generation.trackingEpoch)
            assertEquals(1, f.backend.resets.get())
        }
    }

    @Test fun actualModelReplacementChangesTrackIdentityEvenWithinSameStream() {
        PipelineFixture().use { f ->
            for (ms in listOf(0L, 400L, 800L)) {
                f.send(ms); f.admitted(); f.result()
            }
            val before = f.store.get().generation
            f.analyzer.onObjectTrackerReplaced()
            assertTrue(before.sameImageStream(f.store.get().generation))
            assertFalse(before.sameTrackingEpoch(f.store.get().generation))
            assertNull(f.store.get().objects)
            f.send(1_200); f.admitted()
            assertTrue(f.result().objects.isEmpty())
            assertEquals(2, f.backend.resets.get())
        }
    }

    @Test fun allEpochBoundariesIsolateLateTrackerMutationAndRestartTentative() {
        for (boundary in listOf("stop_start", "source", "rotation", "geometry", "od_restart", "gap")) {
            PipelineFixture().use { f ->
                for (ms in listOf(0L, 400L, 800L)) {
                    f.send(ms); f.admitted()
                    val result = f.result()
                    if (ms == 800L) assertEquals(boundary, 1, result.objects.size)
                }
                val oldEpoch = f.store.get().generation.trackingEpoch
                val gate = PipelineGate()
                f.backend.objectGate = gate
                f.send(1_200); f.admitted()
                assertTrue(gate.entered.await(3, TimeUnit.SECONDS))
                try {
                    when (boundary) {
                        "stop_start" -> {
                            f.analyzer.stop(); f.analyzer.start()
                            f.source = f.analyzer.beginSourceSession()
                        }
                        "source" -> f.source = f.analyzer.beginSourceSession()
                        "od_restart" -> {
                            f.analyzer.configureObjectDetection(false, null)
                            f.analyzer.configureObjectDetection(true, null)
                        }
                    }
                    f.send(
                        if (boundary == "gap") 6_201 else 1_600,
                        if (boundary == "geometry") Bitmap.createBitmap(32, 12, Bitmap.Config.ARGB_8888) else pipelinePixels(),
                        if (boundary == "rotation") 90 else 0,
                    )
                    assertNotEquals(boundary, oldEpoch, f.store.get().generation.trackingEpoch)
                    assertNull(f.store.get().objects)
                    assertEquals(boundary, 1, f.backend.resets.get())
                } finally {
                    gate.release.countDown()
                }
                f.admitted()
                val fresh = f.result()
                assertEquals(boundary, f.store.get().generation.trackingEpoch, fresh.generation.trackingEpoch)
                assertTrue("$boundary must start TENTATIVE", fresh.objects.isEmpty())
                assertEquals(boundary, 2, f.backend.resets.get())
                assertNull("No late A publication for $boundary", f.updates.poll())
            }
        }
    }

    @Test fun exceptionDoesNotRestampAndStillReleasesFrame() {
        PipelineFixture().use { f ->
            f.send(0); f.admitted(); val first = f.result()
            val released = CountDownLatch(1)
            f.backend.throwNext = true
            f.send(400, release = { released.countDown() }); f.admitted()
            f.send(450); f.skippedBeforeAdmission()
            assertTrue(released.await(3, TimeUnit.SECONDS))
            assertSame(first, f.store.get().objects)
        }
    }
}
