package com.algorithmic_alliance.eyeaiapp.camera

import android.graphics.Bitmap
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.inference.MonotonicClock
import com.algorithmic_alliance.eyeaiapp.inference.InferenceTelemetry
import com.algorithmic_alliance.eyeaiapp.inference.InferenceMode
import com.algorithmic_alliance.eyeaiapp.inference.SceneChangeMonitor
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import com.algorithmic_alliance.eyeaiapp.object_detection.ObjectTrackingSession
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntimeState
import com.algorithmic_alliance.eyeaiapp.runtime.withAnalysis
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import uniffi.NativeLib.UniffiDetectedObject

/** Real Bitmap tests on Android; all inference is deliberately fake (no model/native calls). */
class AdaptivePipelineInstrumentationTest {
    private class Gate {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        fun block() {
            entered.countDown()
            check(release.await(5, TimeUnit.SECONDS)) { "Test did not release fake inference" }
        }
    }

    private class Backend : FrameAnalysisBackend {
        @Volatile var ready = true
        val readinessReads = LinkedBlockingQueue<Boolean>()
        override val objectModelReady: Boolean get() = ready.also { readinessReads.put(it) }
        override val maxDepthFrameRate: Int? = null
        var motion: Double? = null
        override fun phoneMotionScore() = motion
        var objectGate: Gate? = null
        var beforeAdmissionGate: Gate? = null
        var depthGate: Gate? = null
        @Volatile var captureDepth = false
        var ocrGate: Gate? = null
        var throwNext = false
        val trackerSteps = AtomicInteger()
        val attempts = LinkedBlockingQueue<Boolean>()
        val inferred = LinkedBlockingQueue<Bitmap>()
        val depthInferred = LinkedBlockingQueue<Bitmap>()
        private val trackingSession = ObjectTrackingSession()
        val resets = AtomicInteger()
        private var evidence = 0
        override fun runObjects(
            frame: Bitmap, trackingEpoch: TrackingEpoch, admit: () -> Boolean,
        ): Array<UniffiDetectedObject>? = trackingSession.run(
            epoch = trackingEpoch,
            ready = { ready },
            admit = {
                beforeAdmissionGate?.also { beforeAdmissionGate = null }?.block()
                admit().also { attempts.put(it) }
            },
            reset = { evidence = 0; resets.incrementAndGet() },
        ) {
            trackerSteps.incrementAndGet()
            inferred.put(frame)
            objectGate?.also { objectGate = null }?.block()
            if (throwNext) { throwNext = false; error("Fake detector failure") }
            // Mutate AFTER the barrier, like runYoloOperation after detector completion.
            evidence++
            if (evidence < 3) emptyArray() else arrayOf(
                UniffiDetectedObject(0f, 0f, 1f, 1f, .5f, .5f, 1f, 1f, 1f, 0, "person", 1),
            )
        }
        override fun runDepth(frame: AnalysisFrame): DepthFrameOutput? {
            if (!captureDepth && depthGate == null) return null
            depthInferred.put(frame.bitmap)
            depthGate?.also { depthGate = null }?.block()
            return DepthFrameOutput(NativeLib.NativeFloatBuffer(256 * 256), 256, 256)
        }
        override suspend fun runOcr(frame: Bitmap): List<TextBoundingBox> {
            ocrGate?.block()
            return emptyList()
        }
    }

    private class Fixture : AutoCloseable {
        val time = AtomicLong()
        val backend = Backend()
        val store = AtomicReference(AnalysisResults())
        val scene = SceneChangeMonitor()
        val telemetry = InferenceTelemetry(MonotonicClock { time.get() })
        val updates = LinkedBlockingQueue<ObjectDetectionSnapshot>()
        val analyzer = FrameAnalyzer(backend, { update ->
            update.results?.objects?.let { updates.put(it) }
        }, MonotonicClock { time.get() }, store, scene, telemetry)
        var source: AnalysisSourceSession
        init {
            analyzer.configureObjectDetection(true, 30.0)
            analyzer.start()
            source = analyzer.beginSourceSession()
        }
        fun send(ms: Long, bitmap: Bitmap = pixels(), rotation: Int = 0, release: () -> Unit = {}): AnalysisFrame {
            time.set(ms * 1_000_000)
            // Remote capture timestamp intentionally unrelated to local arrival time.
            val frame = AnalysisFrame.fromBitmap(bitmap, Long.MAX_VALUE, rotation, release)
            assertTrue(source.submitFrame(frame))
            return frame
        }
        fun attempt(expected: Boolean) = assertEquals(expected, backend.attempts.poll(3, TimeUnit.SECONDS))
        fun result(): ObjectDetectionSnapshot = checkNotNull(updates.poll(3, TimeUnit.SECONDS))
        override fun close() { analyzer.shutdown() }
    }

    @Test fun sceneObservesSkippedFramesAndBurstUsesFreshFrameAcrossAllSourceSinks() {
        // All three production adapters transfer to this exact capability boundary.
        for (sourceName in listOf("CameraX", "Media", "EyeAIVision")) {
            Fixture().use { f ->
                f.send(0); f.attempt(true)
                val first = f.result()
                assertTrue(f.scene.lastCallWasBaselineFrame)
                f.send(50); f.attempt(false)
                assertFalse(f.scene.lastCallWasSampled)
                assertSame(first, f.store.get().objects)
                val burst = pixels(inverted = true)
                f.send(100, burst); f.attempt(true)
                val changed = f.result()
                assertEquals(sourceName, 2, f.backend.trackerSteps.get())
                assertEquals(100_000_000L, changed.frameArrivalNanos)
                assertTrue(f.scene.lastCallWasSampled)
                assertFalse(f.scene.lastCallWasBaselineFrame)
                assertEquals(2L, changed.sequence - first.sequence)
            }
        }
    }

    @Test fun hardCapAndPolicyChangePreserveLastStart() {
        Fixture().use { f ->
            f.analyzer.configureObjectDetection(true, 5.0)
            val epochBeforePolicyChange = f.store.get().generation.trackingEpoch
            f.send(0); f.attempt(true); f.result()
            f.send(100, pixels(true)); f.attempt(false)
            f.analyzer.configureObjectDetection(true, 8.0)
            assertEquals(epochBeforePolicyChange, f.store.get().generation.trackingEpoch)
            f.send(124, pixels(true)); f.attempt(false)
            f.send(125, pixels(true)); f.attempt(true)
            assertEquals(125_000_000L, f.result().inferenceStartedNanos)
            assertEquals(1, f.backend.resets.get())
        }
    }

    @Test fun quietActiveBurstQuietAndRateChangesKeepConfirmedTrackerEvidence() {
        Fixture().use { f ->
            f.send(0); f.attempt(true); f.result()
            assertEquals(InferenceMode.QUIET, f.analyzer.telemetrySnapshot().inferenceMode)
            val epoch = f.store.get().generation.trackingEpoch
            f.backend.motion = 1.0
            f.send(200); f.attempt(true); f.result()
            assertEquals(InferenceMode.ACTIVE, f.analyzer.telemetrySnapshot().inferenceMode)
            f.send(300, pixels(true)); f.attempt(true)
            assertEquals(1, f.result().objects.size)
            assertEquals(InferenceMode.BURST, f.analyzer.telemetrySnapshot().inferenceMode)
            f.backend.motion = null
            f.send(2_000, pixels(true)); f.attempt(true)
            assertEquals(1, f.result().objects.size)
            // The first absent-motion sample starts the existing quiet hold.
            f.send(3_100, pixels(true)); f.attempt(true)
            assertEquals(1, f.result().objects.size)
            assertEquals(InferenceMode.QUIET, f.analyzer.telemetrySnapshot().inferenceMode)
            for ((index, rate) in listOf(15.0, 3.0, 15.0).withIndex()) {
                f.analyzer.configureObjectDetection(true, rate)
                f.send(3_500L + index * 400); f.attempt(true)
                assertEquals(1, f.result().objects.size)
            }
            assertEquals(epoch, f.store.get().generation.trackingEpoch)
            assertEquals(1, f.backend.resets.get())
        }
    }

    @Test fun latestSlotDropsBacklogAndNeverReinfersItsSequence() {
        Fixture().use { f ->
            val gate = Gate(); f.backend.objectGate = gate
            val released = CountDownLatch(3)
            f.send(0, release = { released.countDown() })
            assertTrue(gate.entered.await(3, TimeUnit.SECONDS))
            f.send(400, release = { released.countDown() })
            val newest = pixels()
            f.send(800, newest, release = { released.countDown() })
            gate.release.countDown()
            val first = f.result(); val last = f.result()
            assertEquals(1L, first.sequence)
            assertEquals(3L, last.sequence)
            assertEquals(2, f.backend.trackerSteps.get())
            assertNotNull(f.backend.inferred.poll(3, TimeUnit.SECONDS))
            assertSame(newest, f.backend.inferred.poll(3, TimeUnit.SECONDS))
            // No wake remains for sequence 3, even though the clock permits another admission.
            f.time.set(2_000_000_000)
            assertNull(f.updates.poll(100, TimeUnit.MILLISECONDS))
            f.analyzer.stop()
            assertTrue(released.await(3, TimeUnit.SECONDS))
        }
    }

    @Test fun blockedObjectDetectionDoesNotThrottleDepthAndDepthUsesNewestFrame() {
        Fixture().use { f ->
            val objectGate = Gate()
            val depthGate = Gate()
            f.backend.objectGate = objectGate
            f.backend.depthGate = depthGate
            f.backend.captureDepth = true

            val first = pixels()
            f.send(0, first)
            assertTrue(objectGate.entered.await(3, TimeUnit.SECONDS))
            assertTrue(depthGate.entered.await(3, TimeUnit.SECONDS))
            assertSame(first, f.backend.depthInferred.poll(3, TimeUnit.SECONDS))

            f.send(10, pixels(true))
            val newest = Bitmap.createBitmap(16, 12, Bitmap.Config.ARGB_8888)
            f.send(20, newest)

            // OD is still blocked. Releasing only MiDaS must let it skip the queued middle
            // frame and immediately consume the newest accepted CameraX-style frame.
            depthGate.release.countDown()
            assertSame(newest, f.backend.depthInferred.poll(3, TimeUnit.SECONDS))
            assertNull(f.backend.depthInferred.poll(100, TimeUnit.MILLISECONDS))

            objectGate.release.countDown()
        }
    }

    @Test fun stopDuringNativeCallDiscardsResultAndReleasesConsumer() {
        Fixture().use { f ->
            val gate = Gate(); f.backend.objectGate = gate
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
        Fixture().use { f ->
            val gate = Gate(); f.backend.objectGate = gate
            val released = CountDownLatch(1)
            f.send(0, release = { released.countDown() })
            assertTrue(gate.entered.await(3, TimeUnit.SECONDS))
            val oldSource = f.source
            val oldEpoch = f.store.get().generation.trackingEpoch
            f.analyzer.stop(); f.analyzer.start()
            f.source = f.analyzer.beginSourceSession()
            val rejected = AtomicInteger()
            assertFalse(oldSource.submitFrame(AnalysisFrame.fromBitmap(pixels(), onReleased = { rejected.incrementAndGet() })))
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
        Fixture().use { f ->
            val gate = Gate(); f.backend.objectGate = gate
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
        Fixture().use { f ->
            val objectGate = Gate(); val depthGate = Gate()
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
            f.send(400, pixels(true))
            assertTrue(f.scene.lastCallWasBaselineFrame)
            objectGate.release.countDown(); depthGate.release.countDown()
            assertTrue(released.await(3, TimeUnit.SECONDS))
            assertEquals(400_000_000L, f.result().frameArrivalNanos)
            assertNull(f.store.get().depth)
        }
    }

    @Test fun geometryRotationAndLongGapDiscardVisualEvidenceButPreserveSink() {
        Fixture().use { f ->
            f.send(0); f.attempt(true); f.result()
            f.send(100, pixels(true)); f.attempt(true); f.result()
            val oldGeneration = f.store.get().generation
            val oldEpoch = oldGeneration.trackingEpoch
            f.send(200, rotation = 90); f.attempt(false)
            assertTrue(f.scene.lastCallWasBaselineFrame)
            assertNull(f.store.get().objects)
            assertNotEquals(oldGeneration.content, f.store.get().generation.content)
            assertNotEquals(oldEpoch, f.store.get().generation.trackingEpoch)
            val rotationEpoch = f.store.get().generation.trackingEpoch
            f.send(6_000, rotation = 90); f.attempt(true); f.result()
            assertTrue(f.source.isCurrent())
            assertTrue(f.scene.lastCallWasBaselineFrame)
            assertNotEquals(rotationEpoch, f.store.get().generation.trackingEpoch)
            f.send(6_050, rotation = 90); f.attempt(false)
            val wider = Bitmap.createBitmap(32, 12, Bitmap.Config.ARGB_8888)
            f.send(6_100, wider, 90); f.attempt(false)
            assertTrue(f.scene.lastCallWasBaselineFrame)
        }
    }

    @Test fun lowRateSourceDoesNotResetOnEveryFrame() {
        Fixture().use { f ->
            f.send(0); f.attempt(true); f.result()
            val generation = f.store.get().generation
            f.send(2_000); f.attempt(true); f.result()
            assertFalse(f.scene.lastCallWasBaselineFrame)
            assertEquals(generation, f.store.get().generation)
            assertEquals(generation.trackingEpoch, f.store.get().generation.trackingEpoch)
            assertEquals(1, f.backend.resets.get())
        }
    }

    @Test fun actualModelReplacementChangesTrackIdentityEvenWithinSameStream() {
        Fixture().use { f ->
            for (ms in listOf(0L, 400L, 800L)) {
                f.send(ms); f.attempt(true); f.result()
            }
            val before = f.store.get().generation
            f.analyzer.onObjectTrackerReplaced()
            assertTrue(before.sameImageStream(f.store.get().generation))
            assertFalse(before.sameTrackingEpoch(f.store.get().generation))
            assertNull(f.store.get().objects)
            f.send(1_200); f.attempt(true)
            assertTrue(f.result().objects.isEmpty())
            assertEquals(2, f.backend.resets.get())
        }
    }

    @Test fun allEpochBoundariesIsolateLateTrackerMutationAndRestartTentative() {
        for (boundary in listOf("stop_start", "source", "rotation", "geometry", "od_restart", "gap")) {
            Fixture().use { f ->
                for (ms in listOf(0L, 400L, 800L)) {
                    f.send(ms); f.attempt(true)
                    val result = f.result()
                    if (ms == 800L) assertEquals(boundary, 1, result.objects.size)
                }
                val oldEpoch = f.store.get().generation.trackingEpoch
                val gate = Gate()
                f.backend.objectGate = gate
                f.send(1_200); f.attempt(true)
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
                        if (boundary == "geometry") Bitmap.createBitmap(32, 12, Bitmap.Config.ARGB_8888) else pixels(),
                        if (boundary == "rotation") 90 else 0,
                    )
                    assertNotEquals(boundary, oldEpoch, f.store.get().generation.trackingEpoch)
                    assertNull(f.store.get().objects)
                    // Boundary is non-blocking; reset cannot overtake old native work.
                    assertEquals(boundary, 1, f.backend.resets.get())
                } finally {
                    gate.release.countDown()
                }
                f.attempt(true)
                val fresh = f.result()
                assertEquals(boundary, f.store.get().generation.trackingEpoch, fresh.generation.trackingEpoch)
                assertTrue("$boundary must start TENTATIVE", fresh.objects.isEmpty())
                assertEquals(boundary, 2, f.backend.resets.get())
                assertNull("No late A publication for $boundary", f.updates.poll())
            }
        }
    }

    @Test fun sensorHintAloneCannotProduceBurstAndNullDoesNotRestampResults() {
        Fixture().use { f ->
            f.backend.motion = 1.0
            f.send(0); f.attempt(true); val result = f.result()
            f.send(100); f.attempt(false)
            assertSame(result, f.store.get().objects)
            f.send(143); f.attempt(true); f.result()
            f.backend.motion = null
            f.send(150); f.attempt(false)
        }
    }

    @Test fun exceptionDoesNotRestampAndStillReleasesFrame() {
        Fixture().use { f ->
            f.send(0); f.attempt(true); val first = f.result()
            val released = CountDownLatch(1)
            f.backend.throwNext = true
            f.send(400, release = { released.countDown() }); f.attempt(true)
            f.send(450); f.attempt(false)
            assertTrue(released.await(3, TimeUnit.SECONDS))
            assertSame(first, f.store.get().objects)
        }
    }

    @Test fun ownershipSurvivesConcurrentDepthOcrAndStop() {
        Fixture().use { f ->
            val depthGate = Gate(); val ocrGate = Gate()
            f.backend.depthGate = depthGate; f.backend.ocrGate = ocrGate
            val released = CountDownLatch(1)
            val count = AtomicInteger()
            f.send(0, release = { count.incrementAndGet(); released.countDown() })
            f.attempt(true); f.result()
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
        Fixture().use { f ->
            f.analyzer.stop()
            val count = AtomicInteger()
            val frame = AnalysisFrame.fromBitmap(pixels(), onReleased = { count.incrementAndGet() })
            assertFalse(f.source.submitFrame(frame))
            assertEquals(1, count.get())
            assertFalse(frame.tryRetain())
            assertThrows(IllegalStateException::class.java) { frame.release() }
            assertEquals(1, count.get())
        }
    }

    @Test fun telemetryCountsOnlyRealInferencesAndSchedulerSkips() {
        Fixture().use { f ->
            f.send(0); f.attempt(true); f.result()
            f.send(50); f.attempt(false)
            f.send(100, pixels(inverted = true)); f.attempt(true); f.result()

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
        Fixture().use { f ->
            f.backend.ready = false
            val released = CountDownLatch(1)
            f.send(0, release = { released.countDown() })
            assertEquals(false, f.backend.readinessReads.poll(3, TimeUnit.SECONDS))
            assertNull(f.backend.attempts.poll())
            assertNull(f.store.get().objects)
            f.backend.ready = true
            f.send(50); f.attempt(true)
            assertEquals(50_000_000L, f.result().inferenceStartedNanos)
            assertTrue(released.await(3, TimeUnit.SECONDS))
            assertEquals(1, f.backend.trackerSteps.get())
        }
    }

    @Test fun frameReplacedWhileWaitingForModelLockDoesNotConsumeAdmission() {
        Fixture().use { f ->
            val gate = Gate(); f.backend.beforeAdmissionGate = gate
            val released = CountDownLatch(1)
            f.send(0, release = { released.countDown() })
            assertTrue(gate.entered.await(3, TimeUnit.SECONDS))
            val newest = pixels(true)
            f.send(50, newest)
            gate.release.countDown()
            f.attempt(false); f.attempt(true)
            val result = f.result()
            assertEquals(2L, result.sequence)
            assertEquals(50_000_000L, result.inferenceStartedNanos)
            assertEquals(1, f.backend.trackerSteps.get())
            assertSame(newest, f.backend.inferred.poll())
            assertTrue(released.await(3, TimeUnit.SECONDS))
        }
    }

    @Test fun samplingFailureReleasesInitialReferenceWithoutReplacingLatest() {
        Fixture().use { f ->
            f.send(0); f.attempt(true); val previous = f.result()
            val released = AtomicInteger()
            val bitmap = pixels()
            val frame = AnalysisFrame.fromBitmap(bitmap, onReleased = { released.incrementAndGet() })
            bitmap.recycle()
            f.time.set(100_000_000)
            assertThrows(IllegalArgumentException::class.java) { f.source.submitFrame(frame) }
            assertEquals(1, released.get())
            assertFalse(frame.tryRetain())
            assertSame(previous, f.store.get().objects)
            f.send(400); f.attempt(true)
            assertEquals(previous.sequence + 1, f.result().sequence)
        }
    }

    @Test fun runtimeStateDistinguishesNoUpdateEmptyAndInvalidation() {
        val generation = AnalysisGeneration(1, 2, 3, 4)
        val objects = listOf(UniffiDetectedObject(0f, 0f, 1f, 1f, .5f, .5f, 1f, 1f, 1f, 0, "person", 1))
        val snapshot = ObjectDetectionSnapshot(objects, 0, 0, 0, 1, generation)
        val depth = DepthSnapshot(NativeLib.NativeFloatBuffer(1), 1, 1, 0, 0, generation)
        val results = AnalysisResults(generation, snapshot, depth)
        val preview = pixels()
        val state = EyeAIRuntimeState(analysisResults = results, depthPreviewBitmap = preview)
        assertSame(results, state.withAnalysis(FrameAnalysisUpdate()).analysisResults)
        val empty = state.withAnalysis(FrameAnalysisUpdate(results = results.copy(objects = snapshot.copy(objects = emptyList()))))
        assertNotNull(empty.analysisResults.objects)
        assertTrue(empty.detectedObjects.isEmpty())
        assertSame(preview, empty.depthPreviewBitmap)
        val invalid = state.withAnalysis(FrameAnalysisUpdate(results = AnalysisResults(generation.copy(run = 2))))
        assertNull(invalid.analysisResults.objects)
        assertNull(invalid.depthPreviewBitmap)
        assertTrue(invalid.detectedObjects.isEmpty())
    }

    companion object {
        private fun pixels(inverted: Boolean = false): Bitmap {
            val bitmap = Bitmap.createBitmap(16, 12, Bitmap.Config.ARGB_8888)
            for (y in 0 until 12) for (x in 0 until 16) {
                bitmap.setPixel(x, y, if ((x < 8) xor inverted) -1 else 0xff000000.toInt())
            }
            return bitmap
        }
    }
}
