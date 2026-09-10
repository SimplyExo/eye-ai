package com.algorithmic_alliance.eyeaiapp.camera

import android.graphics.Bitmap
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.inference.throttling.InferenceTelemetry
import com.algorithmic_alliance.eyeaiapp.inference.throttling.MonotonicClock
import com.algorithmic_alliance.eyeaiapp.inference.throttling.SceneChangeMonitor
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import com.algorithmic_alliance.eyeaiapp.object_detection.ObjectTrackingSession
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.fail
import org.junit.Assert.assertTrue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import uniffi.NativeLib.UniffiDetectedObject

internal class PipelineGate {
    val entered = CountDownLatch(1)
    val release = CountDownLatch(1)
    fun block() {
        entered.countDown()
        check(release.await(5, TimeUnit.SECONDS)) { "Test did not release fake inference" }
    }
}

internal class PipelineBackend : FrameAnalysisBackend {
    @Volatile var ready = true
    val readinessReads = LinkedBlockingQueue<Boolean>()
    override val objectModelReady: Boolean get() = ready.also { readinessReads.put(it) }
    override val maxDepthFrameRate: Int? = null
    var motion: Double? = null
    override fun phoneMotionScore() = motion
    var objectGate: PipelineGate? = null
    var beforeAdmissionGate: PipelineGate? = null
    var depthGate: PipelineGate? = null
    @Volatile var captureDepth = false
    var ocrGate: PipelineGate? = null
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

internal class PipelineFixture : AutoCloseable {
    val time = AtomicLong()
    val backend = PipelineBackend()
    val store = AtomicReference(AnalysisResults())
    val scene = SceneChangeMonitor()
    val telemetry = InferenceTelemetry(MonotonicClock { time.get() })
    val updates = LinkedBlockingQueue<ObjectDetectionSnapshot>()
    val analyzer = FrameAnalyzer(backend, { update ->
        update.results?.objects?.let { updates.put(it) }
    }, MonotonicClock { time.get() }, store, scene, telemetry)
    private var observedSchedulerSkips = 0L
    var source: AnalysisSourceSession
    init {
        analyzer.configureObjectDetection(true, 15.0)
        analyzer.start()
        source = analyzer.beginSourceSession()
    }
    fun send(ms: Long, bitmap: Bitmap = pipelinePixels(), rotation: Int = 0, release: () -> Unit = {}): AnalysisFrame {
        time.set(ms * 1_000_000)
        val frame = AnalysisFrame.fromBitmap(bitmap, Long.MAX_VALUE, rotation, release)
        assertTrue(source.submitFrame(frame))
        return frame
    }
    fun admitted() = assertEquals(true, backend.attempts.poll(3, TimeUnit.SECONDS))

    fun rejectedAtAdmission() = assertEquals(false, backend.attempts.poll(3, TimeUnit.SECONDS))

    fun skippedBeforeAdmission() {
        val expected = observedSchedulerSkips + 1
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3)
        while (System.nanoTime() < deadline) {
            val actual = analyzer.telemetrySnapshot().schedulerSkippedFrames
            if (actual >= expected) {
                assertEquals(expected, actual)
                assertNull(backend.attempts.poll())
                observedSchedulerSkips = actual
                return
            }
            Thread.sleep(1)
        }
        fail("Expected scheduler skip $expected, observed $observedSchedulerSkips")
    }

    fun result(): ObjectDetectionSnapshot = checkNotNull(updates.poll(3, TimeUnit.SECONDS))
    override fun close() { analyzer.shutdown() }
}

internal fun pipelinePixels(inverted: Boolean = false): Bitmap {
    val bitmap = Bitmap.createBitmap(16, 12, Bitmap.Config.ARGB_8888)
    for (y in 0 until 12) for (x in 0 until 16) {
        bitmap.setPixel(x, y, if ((x < 8) xor inverted) -1 else 0xff000000.toInt())
    }
    return bitmap
}
