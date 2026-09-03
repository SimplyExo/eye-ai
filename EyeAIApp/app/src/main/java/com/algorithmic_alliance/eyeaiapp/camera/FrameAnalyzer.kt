package com.algorithmic_alliance.eyeaiapp.camera

import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.AIModelData
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntime
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.time.Duration.Companion.seconds
import kotlin.time.measureTime
import uniffi.NativeLib.UniffiDetectedObject

/**
 * Results emitted by the single, source-neutral analysis pipeline.
 *
 * The UI observes these values through [EyeAIRuntime]. No UI callback is held
 * by this class, so a destroyed Activity cannot stop or retain the pipeline.
 */
data class FrameAnalysisUpdate(
    val depthPreviewBitmap: Bitmap? = null,
    val debugInputBitmap: Bitmap? = null,
    val performanceText: String? = null,
    val detectedObjects: Array<UniffiDetectedObject>? = null,
    val frameSize: Size? = null,
)

/**
 * Common frame analyzer used by every future video input source.
 *
 * CameraX is intentionally absent from this class. A source adapter converts
 * its native buffer into [AnalysisFrame] and transfers ownership via
 * [submitFrame]. A future WebRTC adapter can use the same entry point without
 * creating a second inference or model pipeline.
 */
class FrameAnalyzer(
    private val runtime: EyeAIRuntime,
    private val onUpdate: (FrameAnalysisUpdate) -> Unit,
) {
    private val latestFrame = AtomicReference<AnalysisFrame?>(null)
    private val frameSequence = AtomicLong(0L)
    private val frameAvailable = MutableStateFlow(0L)
    private val lifecycleJob = SupervisorJob()
    private val depthExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val objectExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val depthScope = CoroutineScope(lifecycleJob + depthExecutor.asCoroutineDispatcher())
    private val objectScope = CoroutineScope(lifecycleJob + objectExecutor.asCoroutineDispatcher())

    private val stateLock = Any()
    private var depthJob: Job? = null
    private var objectJob: Job? = null
    private var startedValue = false
    private var shutdownValue = false

    @Volatile
    private var formattedSourceFrame = ""

    val started: Boolean
        get() = synchronized(stateLock) { startedValue }

    /** Starts the workers once. It is safe to call again after a source switch. */
    fun start() {
        synchronized(stateLock) {
            if (startedValue || shutdownValue) return
            startedValue = true
            depthJob = depthScope.launch { runDepthLoop() }
            objectJob = objectScope.launch { runObjectDetectionLoop() }
        }
    }

    /** Stops processing and releases the analyzer-owned latest-frame reference. */
    fun stop() {
        val frameToRelease: AnalysisFrame?
        synchronized(stateLock) {
            if (!startedValue) return
            startedValue = false
            depthJob?.cancel()
            objectJob?.cancel()
            depthJob = null
            objectJob = null
            frameToRelease = latestFrame.getAndSet(null)
        }
        frameToRelease?.release()
        AIModelData.detectedObjects.set(emptyArray())
    }

    /** Permanently closes the analyzer. The runtime calls this only at process shutdown. */
    fun shutdown() {
        val frameToRelease: AnalysisFrame?
        synchronized(stateLock) {
            if (shutdownValue) return
            shutdownValue = true
            startedValue = false
            depthJob?.cancel()
            objectJob?.cancel()
            depthJob = null
            objectJob = null
            frameToRelease = latestFrame.getAndSet(null)
        }
        frameToRelease?.release()
        lifecycleJob.cancel()
        depthExecutor.shutdownNow()
        objectExecutor.shutdownNow()
        AIModelData.detectedObjects.set(emptyArray())
    }

    /**
     * Transfers the caller's initial reference to this analyzer. If the
     * analyzer is stopped, ownership is returned by releasing the frame.
     */
    fun submitFrame(frame: AnalysisFrame): Boolean {
        synchronized(stateLock) {
            if (!startedValue || shutdownValue) {
                frame.release()
                return false
            }
            val sequence = frameSequence.incrementAndGet()
            latestFrame.getAndSet(frame)?.release()
            frameAvailable.value = sequence
            return true
        }
    }

    /** Convenience adapter for non-CameraX sources that already own a Bitmap. */
    fun submitBitmap(
        bitmap: Bitmap,
        timestampNanos: Long = System.nanoTime(),
        rotationDegrees: Int = 0,
    ): Boolean = submitFrame(
        AnalysisFrame.fromBitmap(
            bitmap = bitmap,
            timestampNanos = timestampNanos,
            rotationDegrees = rotationDegrees,
        )
    )

    /** Records source timing without making the analyzer depend on a source API. */
    fun recordSourceFrame(timestampNanos: Long) {
        val now = System.nanoTime()
        val previous = lastCameraFrameTimestamp
        if (previous > 0L) {
            val durationNanos = now - previous
            val fps = if (durationNanos > 0L) 1_000_000_000.0 / durationNanos else 0.0
            formattedSourceFrame = String.format(
                Locale.US,
                "Camera Frame: %.2f fps (%d ms), source timestamp=%d\n",
                fps,
                durationNanos / 1_000_000,
                timestampNanos,
            )
        }
        lastCameraFrameTimestamp = now
    }

    private var lastCameraFrameTimestamp: Long = 0L

    private suspend fun runDepthLoop() {
        var observedSequence = 0L
        while (currentCoroutineContext().isActive) {
            observedSequence = frameAvailable.first { it > observedSequence }
            val frame = retainLatestFrame() ?: continue
            try {
                val modelInference = runtime.runDepthInference(frame.bitmap) ?: continue
                val inferenceDuration = measureTime {
                    uniffi.NativeLib.newDepthFrame()
                    AIModelData.depthEstimationData.set(modelInference.prediction)
                    val colorMappedImage = NativeLib.metricDepthColormap(
                        modelInference.prediction.asUniffiWrapper(),
                        modelInference.inputDim,
                    )

                    val performanceText = if (runtime.settings.showProfilingInfo) {
                        val inputResolution = "${frame.width}x${frame.height}"
                        val modelInput =
                            "${modelInference.inputDim.width}x${modelInference.inputDim.height}"
                        "Metric Depth model: ${modelInference.modelName}\n" +
                            "Camera resolution: $inputResolution -> Depth model input: $modelInput\n\n" +
                            "${uniffi.NativeLib.formattedDepthFrame()}\n" +
                            "$formattedSourceFrame\n" +
                            if (runtime.settings.enableObjectDetection) {
                                uniffi.NativeLib.formattedObjectFrame()
                            } else {
                                ""
                            }
                    } else {
                        ""
                    }

                    onUpdate(
                        FrameAnalysisUpdate(
                            depthPreviewBitmap = colorMappedImage,
                            debugInputBitmap = frame.bitmap.takeIf {
                                runtime.settings.showDebugInputBitmap
                            },
                            performanceText = performanceText,
                        )
                    )
                }

                val maxFrameRate = runtime.settings.maxDepthFrameRate
                val minInferenceDuration = maxFrameRate?.let { (1.0 / it).seconds }
                if (minInferenceDuration != null && inferenceDuration < minInferenceDuration) {
                    delay(minInferenceDuration - inferenceDuration)
                }
            } catch (cancelled: kotlinx.coroutines.CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e(EyeAIApp.APP_LOG_TAG, "Depth frame processing failed", error)
            } finally {
                frame.release()
            }
        }
    }

    private suspend fun runObjectDetectionLoop() {
        var observedSequence = 0L
        while (currentCoroutineContext().isActive) {
            observedSequence = frameAvailable.first { it > observedSequence }
            val frame = retainLatestFrame() ?: continue
            try {
                if (!runtime.settings.enableObjectDetection) continue
                val inferenceDuration = measureTime {
                    uniffi.NativeLib.newObjectFrame()
                    val objects = runtime.runObjectInference(frame.bitmap)
                    AIModelData.detectedObjects.set(objects ?: emptyArray())
                    onUpdate(
                        FrameAnalysisUpdate(
                            detectedObjects = objects,
                            frameSize = Size(frame.width, frame.height),
                        )
                    )
                }

                val maxFrameRate = runtime.settings.maxObjectDetectionFrameRate
                val minInferenceDuration = maxFrameRate?.let { (1.0 / it).seconds }
                if (minInferenceDuration != null && inferenceDuration < minInferenceDuration) {
                    delay(minInferenceDuration - inferenceDuration)
                }
            } catch (cancelled: kotlinx.coroutines.CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e(EyeAIApp.APP_LOG_TAG, "Object-detection frame processing failed", error)
            } finally {
                frame.release()
            }
        }
    }

    private fun retainLatestFrame(): AnalysisFrame? {
        val frame = latestFrame.get() ?: return null
        return frame.takeIf { it.tryRetain() }
    }

    suspend fun runOcrAnalysis(): Boolean = withContext(Dispatchers.IO) {
        val frame = retainLatestFrame() ?: return@withContext false
        try {
            Log.d(EyeAIApp.APP_LOG_TAG, "Running on-demand OCR analysis")
            val textBoxes = runtime.runOcrInference(frame.bitmap).toTypedArray()
            AIModelData.ocrBoxes.set(textBoxes)
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "OCR analysis completed successfully, found ${textBoxes.size} text boxes",
            )
            true
        } catch (error: Throwable) {
            Log.e(EyeAIApp.APP_LOG_TAG, "Error during on-demand OCR analysis", error)
            false
        } finally {
            frame.release()
        }
    }
}
