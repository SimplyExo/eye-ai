package com.algorithmic_alliance.eyeaiapp.camera

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.AIModelData
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.rel2abs.DetectionFrame
import com.algorithmic_alliance.eyeaiapp.rel2abs.MetricDepthFrame
import com.algorithmic_alliance.eyeaiapp.rel2abs.Rel2AbsContextFeatures
import com.algorithmic_alliance.eyeaiapp.rel2abs.SegmentationContextFrame
import com.algorithmic_alliance.eyeaiapp.inference.throttling.AdaptiveOdGate
import com.algorithmic_alliance.eyeaiapp.inference.throttling.AnalysisClock
import com.algorithmic_alliance.eyeaiapp.inference.throttling.SceneChangeMonitor
import com.algorithmic_alliance.eyeaiapp.inference.throttling.motion.PhoneMotionLifecycle
import com.algorithmic_alliance.eyeaiapp.inference.throttling.motion.PhoneMotionMonitor
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntime
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import uniffi.NativeLib.UniffiDetectedObject
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds
import kotlin.time.measureTime

/**
 * Results emitted by the single, source-neutral analysis pipeline.
 *
 * The UI observes these values through [EyeAIRuntime]. No UI callback is held
 * by this class, so a destroyed Activity cannot stop or retain the pipeline.
 */
data class FrameAnalysisUpdate(
	val depthPreviewBitmap: Bitmap? = null,
	val debugInputBitmap: Bitmap? = null,
	val debugSegmentationBitmap: Bitmap? = null,
	val performanceText: String? = null,
	val detectedObjects: Array<UniffiDetectedObject>? = null,
	val segmentationOutput: NativeLib.NativeIntBuffer? = null,
	val frameSize: Size? = null,
) {
	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as FrameAnalysisUpdate

		if (depthPreviewBitmap != other.depthPreviewBitmap) return false
		if (debugInputBitmap != other.debugInputBitmap) return false
		if (debugSegmentationBitmap != other.debugSegmentationBitmap) return false
		if (performanceText != other.performanceText) return false
		if (!detectedObjects.contentEquals(other.detectedObjects)) return false
		if (segmentationOutput != other.segmentationOutput) return false
		if (frameSize != other.frameSize) return false

		return true
	}

	override fun hashCode(): Int {
		var result = depthPreviewBitmap?.hashCode() ?: 0
		result = 31 * result + (debugInputBitmap?.hashCode() ?: 0)
		result = 31 * result + (debugSegmentationBitmap?.hashCode() ?: 0)
		result = 31 * result + (performanceText?.hashCode() ?: 0)
		result = 31 * result + (detectedObjects?.contentHashCode() ?: 0)
		result = 31 * result + (segmentationOutput?.hashCode() ?: 0)
		result = 31 * result + (frameSize?.hashCode() ?: 0)
		return result
	}
}

// Lightweight source/output heartbeat used by the runtime safety monitor.
data class FrameAnalyzerHealth(
	val sourceFrameCount: Long,
	val submittedFrameCount: Long,
	val lastAcceptedFrameAtNanos: Long,
	val lastDepthOutputAtNanos: Long,
	val lastObjectOutputAtNanos: Long,
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
	context: Context,
	private val runtime: EyeAIRuntime,
	private val onUpdate: (FrameAnalysisUpdate) -> Unit,
) {
	private val latestFrame = AtomicReference<AnalysisFrame?>(null)
	private val scene: SceneChangeMonitor = SceneChangeMonitor()
	val gate: AdaptiveOdGate = AdaptiveOdGate(
		runtime.settings.maxObjectDetectionFrameRate?.toDouble(),
		AnalysisClock.nowNanos(),
	)
	private val motionLifecycle = PhoneMotionLifecycle(
		createMonitor = { PhoneMotionMonitor(context) },
	)
	private val frameSequence = AtomicLong(0L)
	private val frameAvailable = MutableStateFlow(0L)
	private val sourceFrameCount = AtomicLong(0L)
	private val submittedFrameCount = AtomicLong(0L)
	private val lastAcceptedFrameAtNanos = AtomicLong(0L)
	private val lastDepthOutputAtNanos = AtomicLong(0L)
	private val lastObjectOutputAtNanos = AtomicLong(0L)
	private val lifecycleJob = SupervisorJob()
	private val depthExecutor: ExecutorService = Executors.newSingleThreadExecutor()
	private val objectExecutor: ExecutorService = Executors.newSingleThreadExecutor()
	private val segmentationExecutor: ExecutorService = Executors.newSingleThreadExecutor()
	private val depthScope = CoroutineScope(lifecycleJob + depthExecutor.asCoroutineDispatcher())
	private val objectScope = CoroutineScope(lifecycleJob + objectExecutor.asCoroutineDispatcher())
	private val segmentationScope =
		CoroutineScope(lifecycleJob + segmentationExecutor.asCoroutineDispatcher())

	private val stateLock = Any()
	private var depthJob: Job? = null
	private var objectJob: Job? = null
	private var segmentationJob: Job? = null
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
			clearModelOutputs()
			startedValue = true
			lastAcceptedFrameAtNanos.set(0L)
			depthJob = depthScope.launch { runDepthLoop() }
			objectJob = objectScope.launch { runObjectDetectionLoop() }
			segmentationJob = segmentationScope.launch { runSegmentationLoop() }
		}
		refreshMotionMonitoring()
	}

	/** Stops processing and releases the analyzer-owned latest-frame reference. */
	fun stop() {
		val frameToRelease: AnalysisFrame?
		synchronized(stateLock) {
			if (!startedValue) return
			startedValue = false
			depthJob?.cancel()
			objectJob?.cancel()
			segmentationJob?.cancel()
			depthJob = null
			objectJob = null
			segmentationJob = null
			frameToRelease = latestFrame.getAndSet(null)
		}
		frameToRelease?.release()
		lastAcceptedFrameAtNanos.set(0L)
		clearModelOutputs()
		refreshMotionMonitoring()
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
			segmentationJob?.cancel()
			depthJob = null
			objectJob = null
			segmentationJob = null
			frameToRelease = latestFrame.getAndSet(null)
		}
		frameToRelease?.release()
		lifecycleJob.cancel()
		depthExecutor.shutdownNow()
		objectExecutor.shutdownNow()
		clearModelOutputs()
		refreshMotionMonitoring()
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
			submittedFrameCount.incrementAndGet()
			lastAcceptedFrameAtNanos.set(System.nanoTime())
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
		).also { recordSourceFrame(timestampNanos) }
	)

	/** Records source timing without making the analyzer depend on a source API. */
	fun recordSourceFrame(timestampNanos: Long) {
		sourceFrameCount.incrementAndGet()
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
					val rel2Abs = runtime.runRel2AbsInference(frame.bitmap, modelInference)
					if (rel2Abs != null) {
						AIModelData.rel2AbsFrameCache.publishDepth(
							MetricDepthFrame(
								depthMeters = rel2Abs.depthMeters,
								width = modelInference.inputDim.width,
								height = modelInference.inputDim.height,
								sourceTimestampNanos = frame.timestampNanos,
								sourceWidth = frame.width,
								sourceHeight = frame.height,
								rotationDegrees = frame.rotationDegrees,
								rel2absMode = rel2Abs.mode,
								cameraIntrinsics = frame.cameraIntrinsics,
								neuralGateRunner = runtime.rel2AbsNeuralGateRunner,
							),
						)
					} else {
						AIModelData.rel2AbsFrameCache.clearDepth()
					}
					lastDepthOutputAtNanos.set(System.nanoTime())
					val colorMappedImage = NativeLib.metricDepthColormap(
						modelInference.prediction.asUniffiWrapper(),
						modelInference.inputDim,
					)

					/*
					 *
					 * performanceText.text =
									"Metric Depth model: ${metricDepthModel.name}\nCamera resolution: $formattedInputResolution -> Depth model input: $formattedDepthModelInputSize\n\n${uniffi.NativeLib.formattedDepthFrame()}\n$formattedCameraFrame\n${uniffi.NativeLib.formattedObjectFrame()}\n${uniffi.NativeLib.formattedAudioFrame()}\n${uniffi.NativeLib.formattedDepthAudioThreadFrame()}\n${uniffi.NativeLib.formattedObjectAudioThreadFrame()}"
					 */

					val performanceText = if (runtime.settings.showProfilingInfo) {
						val inputResolution = "${frame.width}x${frame.height}"
						val modelInput =
							"${modelInference.inputDim.width}x${modelInference.inputDim.height}"

						"Metric Depth model: ${modelInference.modelName}\n" + "Camera resolution: $inputResolution -> Depth model input: $modelInput\n" + "Inference mode: ${runtime.frameAnalyzer.gate.mode}\n\n" + "${uniffi.NativeLib.formattedDepthFrame()}\n" + "$formattedSourceFrame\n" + if (runtime.settings.enableObjectDetection) {
							"${uniffi.NativeLib.formattedObjectFrame()}\n"
						} else {
							""
						} + if (runtime.settings.enableSegmentation) {
							"${uniffi.NativeLib.formattedSegmentationFrame()}\n"
						} else {
							""
						} + "${uniffi.NativeLib.formattedAudioFrame()}\n" + "${uniffi.NativeLib.formattedDepthAudioThreadFrame()}\n" + uniffi.NativeLib.formattedObjectAudioThreadFrame()
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
				AIModelData.depthEstimationData.set(null)
				AIModelData.rel2AbsFrameCache.clearDepth()
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
				refreshMotionMonitoring()
				if (!runtime.settings.enableObjectDetection) {
					AIModelData.rel2AbsFrameCache.clearDetections()
					continue
				}

				val now = AnalysisClock.nowNanos()
				gate.updateObjectDetectionBudget(runtime.settings.maxObjectDetectionFrameRate?.toDouble(), now)
				scene.sample(frame.bitmap, frame.rotationDegrees, now)?.let { sample ->
					if (!sample.baselineFrame) gate.onVisualSample(
						sample.score,
						sample.sampledAtNanos
					)
				}
				val decision = gate.tryAcquire(
					phoneMotionScore = motionLifecycle.score(),
					nowNanos = now,
				)
				if (!decision.admitted) {
					val delayNanos = decision.inferenceIntervalNanos
					delay((delayNanos.coerceAtLeast(0L) / 1_000_000).milliseconds)
					continue
				}
				val inferenceDuration = measureTime {
					uniffi.NativeLib.newObjectFrame()
					val objects = runtime.runObjectInference(frame.bitmap)
					val safeObjects = objects ?: emptyArray()
					AIModelData.detectedObjects.set(safeObjects)
					AIModelData.rel2AbsFrameCache.publishDetections(
						DetectionFrame(
							detections = safeObjects,
							sourceTimestampNanos = frame.timestampNanos,
							sourceWidth = frame.width,
							sourceHeight = frame.height,
							rotationDegrees = frame.rotationDegrees,
							objectContextFeatures = Rel2AbsContextFeatures.objectFeatures(safeObjects),
						),
					)
					lastObjectOutputAtNanos.set(System.nanoTime())
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
				AIModelData.detectedObjects.set(emptyArray())
				AIModelData.rel2AbsFrameCache.clearDetections()
				Log.e(EyeAIApp.APP_LOG_TAG, "Object-detection frame processing failed", error)
			} finally {
				frame.release()
			}
		}
	}

	private suspend fun runSegmentationLoop() {
		var observedSequence = 0L
		while (currentCoroutineContext().isActive) {
			observedSequence = frameAvailable.first { it > observedSequence }
			val frame = retainLatestFrame() ?: continue
			try {
				if (!runtime.settings.enableSegmentation) {
					AIModelData.rel2AbsFrameCache.clearSegmentationContexts()
					continue
				}
				val inferenceDuration = measureTime {
					uniffi.NativeLib.newSegmentationFrame()
					val output = runtime.runSegmentationInference(frame.bitmap) ?: continue
					AIModelData.segmentationOutput.set(output.prediction)
					val segmentationContext = Rel2AbsContextFeatures.segmentationFeatureContext(
						output.prediction,
						output.inputDim.width,
						output.inputDim.height,
					)
					AIModelData.rel2AbsFrameCache.publishSegmentationContext(
						SegmentationContextFrame(
							segmentationFeatures = Rel2AbsContextFeatures.segmentationFeatures(output.prediction),
							sourceTimestampNanos = frame.timestampNanos,
							sourceWidth = frame.width,
							sourceHeight = frame.height,
							rotationDegrees = frame.rotationDegrees,
							globalAreaFractions = segmentationContext.globalAreaFractions,
							gridAreaFractions = segmentationContext.gridAreaFractions,
							available = segmentationContext.available,
						),
					)
					val colorMappedImage = NativeLib.segmentationColormap(
						output.prediction.asUniffiWrapper(),
						output.inputDim,
						output.classColors
					)
					onUpdate(
						FrameAnalysisUpdate(
							debugSegmentationBitmap = colorMappedImage,
							segmentationOutput = output.prediction,
							frameSize = Size(frame.width, frame.height),
						)
					)
				}

				val maxFrameRate = runtime.settings.maxSegmentationFrameRate
				val minInferenceDuration = maxFrameRate?.let { (1.0 / it).seconds }
				if (minInferenceDuration != null && inferenceDuration < minInferenceDuration) {
					delay(minInferenceDuration - inferenceDuration)
				}
			} catch (cancelled: kotlinx.coroutines.CancellationException) {
				throw cancelled
			} catch (error: Throwable) {
				AIModelData.segmentationOutput.set(null)
				Log.e(EyeAIApp.APP_LOG_TAG, "Segmentation frame processing failed", error)
			} finally {
				frame.release()
			}
		}
	}

	private fun retainLatestFrame(): AnalysisFrame? {
		val frame = latestFrame.get() ?: return null
		return frame.takeIf { it.tryRetain() }
	}

	fun healthSnapshot(): FrameAnalyzerHealth = FrameAnalyzerHealth(
		sourceFrameCount = sourceFrameCount.get(),
		submittedFrameCount = submittedFrameCount.get(),
		lastAcceptedFrameAtNanos = lastAcceptedFrameAtNanos.get(),
		lastDepthOutputAtNanos = lastDepthOutputAtNanos.get(),
		lastObjectOutputAtNanos = lastObjectOutputAtNanos.get(),
	)

	// Removes data that would otherwise be reused after an input stall.
	fun clearModelOutputs() {
		AIModelData.detectedObjects.set(emptyArray())
		AIModelData.depthEstimationData.set(null)
		AIModelData.segmentationOutput.set(null)
		AIModelData.rel2AbsFrameCache.clear()
		lastDepthOutputAtNanos.set(0L)
		lastObjectOutputAtNanos.set(0L)
	}

	/** Keeps the phone-motion sensor bound only while the gate can use it. */
	private fun refreshMotionMonitoring() {
		val operationActive = synchronized(stateLock) { startedValue && !shutdownValue }
		motionLifecycle.update(
			operationActive = operationActive,
			objectDetectionEnabled = runtime.settings.enableObjectDetection,
			limiterEnabled = runtime.settings.maxObjectDetectionFrameRate != null,
			profilingEnabled = runtime.settings.showProfilingInfo,
		)
	}

	suspend fun runOcrAnalysis(): Boolean = withContext(Dispatchers.IO) {
		if (!runtime.settings.enableOCR) return@withContext false
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
