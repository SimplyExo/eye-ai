package com.algorithmic_alliance.eyeaiapp.camera

import com.algorithmic_alliance.eyeaiapp.inference.throttling.InferenceMode
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class AdaptiveCadenceInstrumentationTest {
    @Test fun sceneObservesSkippedFramesAndBurstUsesFreshFrameAcrossAllSourceSinks() {
        for (sourceName in listOf("CameraX", "Media", "EyeAIVision")) {
            PipelineFixture().use { f ->
                f.send(0); f.admitted()
                val first = f.result()
                assertEquals(0.0, f.analyzer.telemetrySnapshot().visualChangeScore ?: -1.0, 0.0)
                f.send(50); f.skippedBeforeAdmission()
                assertSame(first, f.store.get().objects)
                val burst = pipelinePixels(inverted = true)
                f.send(100, burst); f.admitted()
                val changed = f.result()
                assertEquals(sourceName, 2, f.backend.trackerSteps.get())
                assertEquals(100_000_000L, changed.frameArrivalNanos)
                assertTrue(f.analyzer.telemetrySnapshot().visualChangeScore!! >= 0.8)
                assertEquals(2L, changed.sequence - first.sequence)
            }
        }
    }

    @Test fun hardCapAndPolicyChangePreserveLastStart() {
        PipelineFixture().use { f ->
            f.analyzer.configureObjectDetection(true, 5.0)
            val epochBeforePolicyChange = f.store.get().generation.trackingEpoch
            f.send(0); f.admitted(); f.result()
            f.send(100, pipelinePixels(true)); f.skippedBeforeAdmission()
            f.analyzer.configureObjectDetection(true, 8.0)
            assertEquals(epochBeforePolicyChange, f.store.get().generation.trackingEpoch)
            f.send(124, pipelinePixels(true)); f.skippedBeforeAdmission()
            f.send(125, pipelinePixels(true)); f.admitted()
            assertEquals(125_000_000L, f.result().inferenceStartedNanos)
            assertEquals(1, f.backend.resets.get())
        }
    }

    @Test fun quietActiveBurstQuietAndRateChangesKeepConfirmedTrackerEvidence() {
        PipelineFixture().use { f ->
            f.send(0); f.admitted(); f.result()
            assertEquals(InferenceMode.QUIET, f.analyzer.telemetrySnapshot().inferenceMode)
            val epoch = f.store.get().generation.trackingEpoch
            f.backend.motion = 1.0
            f.send(200); f.admitted(); f.result()
            assertEquals(InferenceMode.ACTIVE, f.analyzer.telemetrySnapshot().inferenceMode)
            f.send(300, pipelinePixels(true)); f.admitted()
            assertEquals(1, f.result().objects.size)
            assertEquals(InferenceMode.BURST, f.analyzer.telemetrySnapshot().inferenceMode)
            f.backend.motion = null
            f.send(2_000, pipelinePixels(true)); f.admitted()
            assertEquals(1, f.result().objects.size)
            f.send(3_100, pipelinePixels(true)); f.admitted()
            assertEquals(1, f.result().objects.size)
            assertEquals(InferenceMode.QUIET, f.analyzer.telemetrySnapshot().inferenceMode)
            for ((index, rate) in listOf(15.0, 3.0, 15.0).withIndex()) {
                f.analyzer.configureObjectDetection(true, rate)
                f.send(3_500L + index * 400); f.admitted()
                assertEquals(1, f.result().objects.size)
            }
            assertEquals(epoch, f.store.get().generation.trackingEpoch)
            assertEquals(1, f.backend.resets.get())
        }
    }

    @Test fun sensorHintAloneCannotProduceBurstAndNullDoesNotRestampResults() {
        PipelineFixture().use { f ->
            f.backend.motion = 1.0
            f.send(0); f.admitted(); val result = f.result()
            assertEquals(InferenceMode.ACTIVE, f.analyzer.telemetrySnapshot().inferenceMode)
            f.send(100); f.skippedBeforeAdmission()
            assertSame(result, f.store.get().objects)
            f.send(143); f.admitted(); val refreshed = f.result()
            f.backend.motion = null
            f.send(150); f.skippedBeforeAdmission()
            assertSame(refreshed, f.store.get().objects)
            assertNotEquals(InferenceMode.BURST, f.analyzer.telemetrySnapshot().inferenceMode)
        }
    }

    @Test fun freshSensorMotionRaisesCadenceBeforeWaitingForTheModelLock() {
        PipelineFixture().use { f ->
            f.send(0); f.admitted(); f.result()
            f.backend.motion = 1.0

            f.send(110); f.admitted(); f.result()
            assertEquals(InferenceMode.ACTIVE, f.analyzer.telemetrySnapshot().inferenceMode)
        }
    }

    @Test fun telemetryCountsOnlyRealInferencesAndGateSkips() {
        PipelineFixture().use { f ->
            f.send(0); f.admitted(); f.result()
            f.send(50); f.skippedBeforeAdmission()
            f.send(100, pipelinePixels(inverted = true)); f.admitted(); f.result()

            val snapshot = f.analyzer.telemetrySnapshot()
            assertEquals(0.0, snapshot.phoneMotionScore ?: 0.0, 0.0)
            assertEquals(1L, snapshot.schedulerSkippedFrames)
            assertEquals(2L, snapshot.objectInferenceCount)
            assertEquals(100_000_000L, snapshot.lastInferenceIntervalNanos)
            assertNotNull(snapshot.lastInferenceRuntimeNanos)
            assertEquals("strong_visual_change", snapshot.modeChangeReason)
        }
    }

    @Test fun notReadyConsumesSequenceButNoAdmissionOrResult() {
        PipelineFixture().use { f ->
            f.backend.ready = false
            val released = CountDownLatch(1)
            f.send(0, release = { released.countDown() })
            assertEquals(false, f.backend.readinessReads.poll(3, TimeUnit.SECONDS))
            assertNull(f.backend.attempts.poll())
            assertNull(f.store.get().objects)
            f.backend.ready = true
            f.send(50); f.admitted()
            assertEquals(50_000_000L, f.result().inferenceStartedNanos)
            assertTrue(released.await(3, TimeUnit.SECONDS))
            assertEquals(1, f.backend.trackerSteps.get())
        }
    }

    @Test fun frameReplacedWhileWaitingForModelLockDoesNotConsumeAdmission() {
        PipelineFixture().use { f ->
            val gate = PipelineGate(); f.backend.beforeAdmissionGate = gate
            val released = CountDownLatch(1)
            f.send(0, release = { released.countDown() })
            assertTrue(gate.entered.await(3, TimeUnit.SECONDS))
            val newest = pipelinePixels(true)
            f.send(50, newest)
            gate.release.countDown()
            f.rejectedAtAdmission(); f.admitted()
            val result = f.result()
            assertEquals(2L, result.sequence)
            assertEquals(50_000_000L, result.inferenceStartedNanos)
            assertEquals(1, f.backend.trackerSteps.get())
            assertSame(newest, f.backend.inferred.poll())
            assertTrue(released.await(3, TimeUnit.SECONDS))
        }
    }
}
