package com.algorithmic_alliance.eyeaiapp.camera

import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.AIModelData
import com.algorithmic_alliance.eyeaiapp.inference.AnalysisClock
import com.algorithmic_alliance.eyeaiapp.inference.InferenceScheduler
import com.algorithmic_alliance.eyeaiapp.inference.InferenceTelemetry
import com.algorithmic_alliance.eyeaiapp.inference.InferenceTelemetrySnapshot
import com.algorithmic_alliance.eyeaiapp.inference.MonotonicClock
import com.algorithmic_alliance.eyeaiapp.inference.ObjectDetectionV1Policy
import com.algorithmic_alliance.eyeaiapp.inference.SceneChangeMonitor
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import java.util.Locale
import kotlin.time.Duration.Companion.seconds
import kotlin.time.Duration.Companion.nanoseconds
import kotlin.time.measureTime
import uniffi.NativeLib.UniffiDetectedObject

data class FrameAnalysisUpdate(
    val depthPreviewBitmap: Bitmap? = null,
    val debugInputBitmap: Bitmap? = null,
    val performanceText: String? = null,
    /** null = no result update; a non-null value can explicitly clear either result. */
    val results: AnalysisResults? = null,
    val frameSize: Size? = null,
)

data class DepthFrameOutput(
    val prediction: com.algorithmic_alliance.eyeaiapp.NativeLib.NativeFloatBuffer,
    val width: Int,
    val height: Int,
    val presentation: FrameAnalysisUpdate = FrameAnalysisUpdate(),
    val postProcessingNanos: Long = 0,
)

/** Small model seam: native work and model locks are always outside the analyzer lock. */
interface FrameAnalysisBackend {
    val objectModelReady: Boolean
    val maxDepthFrameRate: Int?
    fun phoneMotionScore(): Double?
    /** Invoke admit after obtaining the model lock/readiness, immediately before actual work. */
    fun runObjects(frame: Bitmap, trackingEpoch: TrackingEpoch, admit: () -> Boolean): Array<UniffiDetectedObject>?
    fun runDepth(frame: AnalysisFrame): DepthFrameOutput?
    suspend fun runOcr(frame: Bitmap): List<TextBoundingBox>
}

/** An immutable capability captured when a source is bound; never refreshed by a callback. */
class AnalysisSourceSession internal constructor(
    private val analyzer: FrameAnalyzer,
    internal val runGeneration: Long,
    internal val sourceGeneration: Long,
) {
    fun submitFrame(frame: AnalysisFrame): Boolean = analyzer.submitFrame(frame, this)

    fun submitBitmap(
        bitmap: Bitmap,
        timestampNanos: Long = AnalysisClock.nowNanos(),
        rotationDegrees: Int = 0,
    ): Boolean = submitFrame(AnalysisFrame.fromBitmap(bitmap, timestampNanos, rotationDegrees))

    fun isCurrent(): Boolean = analyzer.isCurrentSource(this)
}

/**
 * One latest slot and one serial worker per model. onUpdate must be a short, non-blocking state
 * publication; it runs under stateLock so clearing and late result publication cannot reorder.
 */
class FrameAnalyzer(
    private val backend: FrameAnalysisBackend,
    private val onUpdate: (FrameAnalysisUpdate) -> Unit,
    private val clock: MonotonicClock = AnalysisClock,
    private val resultStore: AtomicReference<AnalysisResults> = AIModelData.analysisResults,
    private val scene: SceneChangeMonitor = SceneChangeMonitor(),
    private val telemetry: InferenceTelemetry = InferenceTelemetry(clock),
    private val profilingEnabled: () -> Boolean = { false },
) {
    private data class FrameSlot(
        val frame: AnalysisFrame,
        val sequence: Long,
        val arrivalNanos: Long,
        val arrivalIntervalNanos: Long?,
        val generation: AnalysisGeneration,
    )

    private val stateLock = Any()
    private var latestFrame: FrameSlot? = null
    private var sequence = 0L
    private val frameAvailable = MutableStateFlow(0L)
    private val lifecycleJob = SupervisorJob()
    private val depthExecutor = Executors.newSingleThreadExecutor()
    private val objectExecutor = Executors.newSingleThreadExecutor()
    private val depthScope = CoroutineScope(lifecycleJob + depthExecutor.asCoroutineDispatcher())
    private val objectScope = CoroutineScope(lifecycleJob + objectExecutor.asCoroutineDispatcher())
    private var depthJob: Job? = null
    private var objectJob: Job? = null
    private var startedValue = false
    private var shutdownValue = false
    private var odEnabled = false
    private var generation = AnalysisGeneration()
    private var sourceSession: AnalysisSourceSession? = null
    private var lastArrivalNanos: Long? = null
    private var geometry: Triple<Int, Int, Int>? = null
    private val scheduler = InferenceScheduler(ObjectDetectionV1Policy.config(), clock)

    val started: Boolean get() = synchronized(stateLock) { startedValue }

    fun start() = synchronized(stateLock) {
        if (startedValue || shutdownValue) return@synchronized
        // A new runtime run is a tracking boundary. Cadence state is reset
        // below, but cadence changes themselves never change this generation.
        generation = generation.copy(run = generation.run + 1, objectDetection = generation.objectDetection + 1)
        startedValue = true
        telemetry.startSession(clock.nowNanos())
        resetStreamLocked("start")
        clearResultsLocked()
        val run = generation.run
        val initialSequence = sequence
        depthJob = depthScope.launch { runDepthLoop(run, initialSequence) }
        objectJob = objectScope.launch { runObjectLoop(run, initialSequence) }
    }

    fun stop() = synchronized(stateLock) {
        if (!startedValue) return@synchronized
        startedValue = false
        // Stop invalidates the old run; the next start receives a new epoch.
        generation = generation.copy(run = generation.run + 1, objectDetection = generation.objectDetection + 1)
        sourceSession = null
        depthJob?.cancel()
        objectJob?.cancel()
        depthJob = null
        objectJob = null
        releaseLatestLocked()
        resetStreamLocked("stop")
        clearResultsLocked()
    }

    fun shutdown() {
        synchronized(stateLock) {
            if (shutdownValue) return
            stop()
            shutdownValue = true
        }
        lifecycleJob.cancel()
        // Cooperative cancellation only: never interrupt an in-flight synchronous native call.
        depthExecutor.shutdown()
        objectExecutor.shutdown()
    }

    fun configureObjectDetection(enabled: Boolean, maxRateHz: Double?) = synchronized(stateLock) {
        scheduler.updateConfig(ObjectDetectionV1Policy.config(maxRateHz))
        if (odEnabled != enabled) {
            odEnabled = enabled
            // Only an OD enable/disable transition is a tracking boundary.
            // Changing the configured FPS policy while OD stays enabled is not.
            generation = generation.copy(objectDetection = generation.objectDetection + 1)
            scheduler.resetActivity()
            scene.reset()
            telemetry.resetActivity(
                if (enabled) "object_detection_enabled" else "object_detection_disabled",
                clock.nowNanos(),
            )
            // An enable starts with the next incoming frame, not an old disabled-slot frame.
            clearResultsLocked(objectsOnly = true)
        }
    }

    /**
     * A real model replacement also recreates the native tracker. Called under
     * the YOLO session lock, after native locks have been released. Only takes
     * stateLock and publishes state; never calls back into a model or native code.
     */
    fun onObjectTrackerReplaced() = synchronized(stateLock) {
        generation = generation.copy(objectDetection = generation.objectDetection + 1)
        scheduler.resetActivity()
        scene.reset()
        telemetry.resetActivity("object_tracker_replaced", clock.nowNanos())
        clearResultsLocked(objectsOnly = true)
    }

    fun beginSourceSession(): AnalysisSourceSession = synchronized(stateLock) {
        check(startedValue && !shutdownValue) { "Analyzer must be started before binding a source" }
        // Binding a different source starts a distinct tracking stream.
        generation = generation.copy(source = generation.source + 1)
        releaseLatestLocked()
        resetStreamLocked("source_change")
        clearResultsLocked()
        AnalysisSourceSession(this, generation.run, generation.source).also { sourceSession = it }
    }

    /** Invalidates callbacks immediately, including while a service/source stop is still pending. */
    fun invalidateSource() = synchronized(stateLock) {
        // Invalidation is itself a source boundary, even before a replacement
        // source session is bound.
        generation = generation.copy(source = generation.source + 1)
        sourceSession = null
        releaseLatestLocked()
        resetStreamLocked("source_invalidated")
        clearResultsLocked()
    }

    internal fun isCurrentSource(session: AnalysisSourceSession): Boolean = synchronized(stateLock) {
        startedValue && !shutdownValue && sourceSession === session &&
            session.runGeneration == generation.run && session.sourceGeneration == generation.source
    }

    /** Always consumes the initial reference, including rejection and sampling failure. */
    fun submitFrame(frame: AnalysisFrame, session: AnalysisSourceSession): Boolean = synchronized(stateLock) {
        var transferred = false
        try {
            if (!isCurrentSource(session)) return@synchronized false
            val now = clock.nowNanos()
            val nextGeometry = Triple(frame.width, frame.height, Math.floorMod(frame.rotationDegrees, 360))
            val gap = lastArrivalNanos?.let { now - it > ObjectDetectionV1Policy.STREAM_GAP_NANOS } == true
            if (gap || (geometry != null && geometry != nextGeometry)) {
                // A long gap or a representation/rotation change is a content
                // boundary. Ordinary scheduler skips do not reach this path.
                generation = generation.copy(content = generation.content + 1)
                scheduler.resetActivity()
                scene.reset()
                telemetry.resetActivity(
                    if (gap) "stream_gap" else "frame_geometry_changed",
                    now,
                )
                clearResultsLocked()
            }
            val arrivalInterval = lastArrivalNanos?.let { now - it }?.takeIf { it > 0 && !gap }
            lastArrivalNanos = now
            geometry = nextGeometry
            // The initial reference is still ours here. Only 16x12 pixels are sampled.
            val score = scene.update(frame.bitmap, frame.rotationDegrees, now)
            if (scene.lastCallWasSampled) {
                val sampledAt = checkNotNull(scene.lastSampledAtNanos)
                telemetry.recordVisualChange(score, sampledAt)
                if (!scene.lastCallWasBaselineFrame) {
                    val schedulerSnapshot = scheduler.updateVisualChange(score, atNanos = sampledAt)
                    telemetry.observeScheduler(schedulerSnapshot, score, null, sampledAt)
                }
            }
            val old = latestFrame
            latestFrame = FrameSlot(frame, ++sequence, now, arrivalInterval, generation)
            transferred = true
            old?.frame?.release()
            frameAvailable.value = sequence
            true
        } finally {
            if (!transferred) frame.release()
        }
    }

    private fun resetStreamLocked(reason: String) {
        lastArrivalNanos = null
        geometry = null
        scene.reset()
        scheduler.resetActivity()
        telemetry.resetActivity(reason, clock.nowNanos())
    }

    private fun releaseLatestLocked() {
        val old = latestFrame
        latestFrame = null
        old?.frame?.release()
    }

    private fun clearResultsLocked(objectsOnly: Boolean = false) {
        val next = AnalysisResults(
            generation = generation,
            depth = resultStore.get().depth.takeIf { objectsOnly },
        )
        resultStore.set(next)
        if (!objectsOnly) AIModelData.ocrBoxes.set(null)
        onUpdate(FrameAnalysisUpdate(results = next))
    }

    /** Frame, sequence, arrival and generation are acquired together, never from the wake value. */
    private fun retainLatest(run: Long, observed: Long): Pair<FrameSlot?, Long> = synchronized(stateLock) {
        val slot = if (!startedValue || generation.run != run) null
        else latestFrame?.takeIf { it.sequence > observed && it.frame.tryRetain() }
        // Even the empty-slot case observes sequence under the same lock. Otherwise a frame
        // arriving between a null take and a separate sequence read could be silently skipped.
        slot to (slot?.sequence ?: sequence)
    }

    private fun validStream(slot: FrameSlot): Boolean =
        startedValue && !shutdownValue && slot.generation.sameImageStream(generation)

    private fun validObjects(slot: FrameSlot): Boolean =
        odEnabled && validStream(slot) && slot.generation == generation

    private suspend fun runObjectLoop(run: Long, initialSequence: Long) {
        var observed = initialSequence
        while (currentCoroutineContext().isActive) {
            frameAvailable.first { it > observed }
            val (slot, takenSequence) = retainLatest(run, observed)
            observed = takenSequence // Also consumed on skip, not-ready and exception.
            if (slot == null) continue
            try {
                if (!synchronized(stateLock) { validObjects(slot) } || !backend.objectModelReady) continue
                var admittedAt: Long? = null
                val objects = try {
                    backend.runObjects(slot.frame.bitmap, slot.generation.trackingEpoch) {
                        synchronized(stateLock) {
                            // A newer frame may have arrived while the model lock was busy.
                            if (!validObjects(slot) || latestFrame?.sequence != slot.sequence) {
                                false
                            } else {
                                val motion = backend.phoneMotionScore()
                                val now = clock.nowNanos()
                                telemetry.recordPhoneMotion(motion, now)
                                scheduler.updatePhoneMotion(motion, atNanos = now)
                                val admitted = scheduler.tryAcquireInference(now)
                                val decision = scheduler.snapshot(atNanos = now)
                                telemetry.observeScheduler(decision, scene.lastScore, motion, now)
                                if (!admitted) {
                                    telemetry.recordSchedulerSkip(now)
                                } else {
                                    admittedAt = now
                                    telemetry.recordInferenceStart(now)
                                }
                                admitted
                            }
                        }
                    }
                } finally {
                    admittedAt?.let { telemetry.recordInferenceCompletion(it, clock.nowNanos()) }
                }
                maybeLogTelemetry()
                if (objects == null) continue
                val startedAt = admittedAt ?: continue
                synchronized(stateLock) {
                    if (validObjects(slot)) {
                        val snapshot = ObjectDetectionSnapshot(
                            objects.toList(), slot.arrivalNanos, startedAt, clock.nowNanos(),
                            slot.sequence, slot.generation,
                        )
                        val next = resultStore.get().copy(objects = snapshot)
                        resultStore.set(next)
                        onUpdate(FrameAnalysisUpdate(results = next, frameSize = Size(slot.frame.width, slot.frame.height)))
                    }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e("EyeAI", "Object-detection frame processing failed", error)
                maybeLogTelemetry()
            } finally {
                slot.frame.release()
            }
        }
    }

    private suspend fun runDepthLoop(run: Long, initialSequence: Long) {
        var observed = initialSequence
        while (currentCoroutineContext().isActive) {
            frameAvailable.first { it > observed }
            val (slot, takenSequence) = retainLatest(run, observed)
            observed = takenSequence
            if (slot == null) continue
            try {
                if (!synchronized(stateLock) { validStream(slot) }) continue
                val output = backend.runDepth(slot.frame) ?: continue
                // Preserve the existing independent depth post-processing rate limiter.
                val postProcessingDuration = measureTime {
                    val presentation = output.presentation.let { update ->
                        val interval = slot.arrivalIntervalNanos
                        if (interval == null || update.performanceText.isNullOrEmpty()) update
                        else update.copy(performanceText = update.performanceText + "\n" + String.format(
                            Locale.US, "Camera Frame: %.2f fps (%d ms), source timestamp=%d\n",
                            1_000_000_000.0 / interval, interval / 1_000_000, slot.frame.timestampNanos,
                        ))
                    }
                    synchronized(stateLock) {
                        if (validStream(slot)) {
                            val snapshot = DepthSnapshot(
                                output.prediction, output.width, output.height, slot.arrivalNanos,
                                clock.nowNanos(), slot.generation,
                            )
                            val next = resultStore.get().copy(depth = snapshot)
                            resultStore.set(next)
                            onUpdate(presentation.copy(results = next))
                        }
                    }
                }
                backend.maxDepthFrameRate?.let {
                    val minimum = (1.0 / it).seconds
                    val elapsed = output.postProcessingNanos.nanoseconds + postProcessingDuration
                    if (elapsed < minimum) delay(minimum - elapsed)
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e("EyeAI", "Depth frame processing failed", error)
            } finally {
                slot.frame.release()
            }
        }
    }

    suspend fun runOcrAnalysis(): Boolean = withContext(Dispatchers.IO) {
        val slot = synchronized(stateLock) {
            latestFrame?.takeIf { startedValue && it.frame.tryRetain() }
        } ?: return@withContext false
        try {
            val boxes = backend.runOcr(slot.frame.bitmap).toTypedArray()
            synchronized(stateLock) {
                if (!validStream(slot)) false
                else {
                    AIModelData.ocrBoxes.set(boxes)
                    true
                }
            }
        } catch (error: Throwable) {
            Log.e("EyeAI", "OCR analysis failed", error)
            false
        } finally {
            slot.frame.release()
        }
    }

    /** A coherent copy for diagnostics and future benchmark consumers. */
    fun telemetrySnapshot(): InferenceTelemetrySnapshot = telemetry.snapshot(clock.nowNanos())

    private fun maybeLogTelemetry() {
        if (!profilingEnabled()) return
        telemetry.pollLogSnapshot(clock.nowNanos())?.let { snapshot ->
            Log.d("EyeAI", "[InferenceTelemetry] ${snapshot.asLogLine()}")
        }
    }
}
