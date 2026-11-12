package com.algorithmic_alliance.eyeaiapp.camera

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.context.ContextSnapshot
import kotlinx.serialization.json.*
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
import com.algorithmic_alliance.eyeaiapp.worldstate.Availability
import com.algorithmic_alliance.eyeaiapp.worldstate.FrameStamp
import com.algorithmic_alliance.eyeaiapp.worldstate.WorldStatePipeline
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
import kotlin.time.Duration.Companion.nanoseconds
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
	val worldState = WorldStatePipeline()
    private var worldStateActive = false
	private class AcceptedFrame(val frame: AnalysisFrame, val stamp: FrameStamp, val capture: ContextSnapshot? = null) {
		val bitmap get() = frame.bitmap
		val width get() = frame.width
		val height get() = frame.height
		val timestampNanos get() = frame.timestampNanos
		val rotationDegrees get() = frame.rotationDegrees
		val cameraIntrinsics get() = frame.cameraIntrinsics
		fun release() = frame.release()
	}
	private val latestFrame = AtomicReference<AcceptedFrame?>(null)
    private val contextDepth=AtomicReference<AcceptedFrame?>(null)
    private val contextObjects=AtomicReference<AcceptedFrame?>(null)
    private val contextSegmentation=AtomicReference<AcceptedFrame?>(null)
    private val contextNarrator get()=runtime.localSceneDescriptions
    private fun channelStatus(frame: AcceptedFrame,channel: String,status: String) {
        if(frame.capture!=null)contextNarrator.capturedStatus(frame.stamp,channel,status)
        contextNarrator.recorder.event("CHANNEL_STATUS",frame=frame.stamp) { buildJsonObject {put("channel",channel)
        put("status",status)} }
    }
    private fun queuedFrame(queue: AtomicReference<AcceptedFrame?>): AcceptedFrame? = synchronized(stateLock) {
        queue.getAndSet(null) ?: retainLatestFrame()
    }
    private fun pin(frame: AcceptedFrame,capture: ContextSnapshot) {
        for(queue in listOf(contextDepth,contextObjects,contextSegmentation)) {
            if(frame.frame.tryRetain())queue.getAndSet(AcceptedFrame(frame.frame,frame.stamp,capture))?.release()
        }
        contextNarrator.recorder.event("CAPTURE_CONFIGURATION",frame=frame.stamp) { buildJsonObject {
            put("episode",capture.episodeId)
put("depthModel",runtime.settings.depthModel)
            put("rel2absMode",runtime.settings.rel2AbsMode.name)
            put("cameraHeightTenths",runtime.settings.cameraHeightTenths)
            put("objectsEnabled",runtime.settings.enableObjectDetection)
            put("segmentationEnabled",runtime.settings.enableSegmentation)
put("fxPx",frame.cameraIntrinsics?.fxPx)
            put("fyPx",frame.cameraIntrinsics?.fyPx)
        } }
    }
    private fun releaseContextFrames() {listOf(contextDepth,contextObjects,contextSegmentation).forEach {it.getAndSet(null)?.release()}}
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
		val frameToRelease: AcceptedFrame?
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
            releaseContextFrames()
		}
		frameToRelease?.release()
		lastAcceptedFrameAtNanos.set(0L)
		clearModelOutputs()
		refreshMotionMonitoring()
	}

	/** Permanently closes the analyzer. The runtime calls this only at process shutdown. */
	fun shutdown() {
		val frameToRelease: AcceptedFrame?
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
            releaseContextFrames()
		}
		frameToRelease?.release()
		lifecycleJob.cancel()
		depthExecutor.shutdownNow()
		objectExecutor.shutdownNow()
		segmentationExecutor.shutdownNow()
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
			val observing = contextNarrator.observing
			if (worldStateActive != observing) {
				// A new session rejects in-flight results from before the transition.
				worldState.analyzer.reset()
				worldStateActive = observing
			}
			val sequence = frameSequence.incrementAndGet()
			val stamp = worldState.analyzer.acceptFrame(frame.timestampNanos, frame.width, frame.height,
				frame.rotationDegrees, frame.timeOrigin)
			submittedFrameCount.incrementAndGet()
			lastAcceptedFrameAtNanos.set(System.nanoTime())
			val accepted=AcceptedFrame(frame,stamp)
            // Observe before publication so all workers see the same priority frame atomically.
            runtime.observeContextFrame(frame.bitmap,stamp,motionLifecycle.score())?.let {pin(accepted,it)}
            latestFrame.getAndSet(accepted)?.release()
			frameAvailable.value = sequence
			return true
		}
	}

	/** Convenience adapter for non-CameraX sources that already own a Bitmap. */
	fun submitBitmap(
		bitmap: Bitmap,
		timestampNanos: Long = System.nanoTime(),
	): Boolean = submitFrame(
		AnalysisFrame.fromBitmap(
			bitmap = bitmap,
			timestampNanos = timestampNanos,
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
			val frame = queuedFrame(contextDepth) ?: continue
			try {
				contextNarrator.recorder.event("DEPTH_MODEL_START",frame=frame.stamp)
                val modelInference = runtime.runDepthInference(frame.bitmap)
                contextNarrator.recorder.event("DEPTH_MODEL_END",frame=frame.stamp)
				if (modelInference == null) {
                    channelStatus(frame,"depth","UNSUPPORTED")
					worldState.analyzer.invalidateDepth(frame.stamp)
					continue
				}
				val inferenceDuration = measureTime {
					uniffi.NativeLib.newDepthFrame()
					AIModelData.depthEstimationData.set(modelInference.prediction)
					contextNarrator.recorder.event("REL2ABS_MODEL_START",frame=frame.stamp)
                    val rel2Abs = runtime.runRel2AbsInference(frame.bitmap, modelInference)
                    contextNarrator.recorder.event("REL2ABS_MODEL_END",frame=frame.stamp)
					val metricFrame = if (rel2Abs != null) {
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
						)
					} else {
						AIModelData.rel2AbsFrameCache.clearDepth()
						null
					}
					metricFrame?.let { AIModelData.rel2AbsFrameCache.publishDepth(it) }
					if (contextNarrator.observing) worldState.depth(frame.stamp, modelInference.rawRelativeDepth,
						modelInference.inputDim.width, modelInference.inputDim.height, modelInference.modelName,
						modelInference.modelName == EyeAIApp.DEFAULT_DEPTH_MODEL_NAME, metricFrame)
                    contextNarrator.recorder.event("DEPTH_METADATA",frame=frame.stamp) { buildJsonObject {
                        put("model",modelInference.modelName)
                        put("rel2absMode",metricFrame?.rel2absMode?.name)
                        put("relativeRepresentationSupported",modelInference.modelName==EyeAIApp.DEFAULT_DEPTH_MODEL_NAME)
                        put("completedAtNanos",System.nanoTime())
                    } }
                    val recording=contextNarrator.recorder.state.value.active
                    val raw=if(recording||frame.capture!=null)FloatArray(modelInference.rawRelativeDepth.floatBuffer.capacity()) {modelInference.rawRelativeDepth.floatBuffer.get(it)}else null
                    if(recording) {
                        contextNarrator.recorder.floats(frame.stamp,"midas-raw",modelInference.inputDim.width,modelInference.inputDim.height,raw!!)
                        contextNarrator.recorder.floats(frame.stamp,"legacy-depth",modelInference.inputDim.width,modelInference.inputDim.height,
                            FloatArray(modelInference.prediction.floatBuffer.capacity()) {modelInference.prediction.floatBuffer.get(it)})
                        metricFrame?.let {contextNarrator.recorder.floats(frame.stamp,"rel2abs-visual",it.width,it.height,it.depthMeters)}
                    }
                    if(frame.capture!=null)contextNarrator.capturedDepth(frame.stamp,raw?.takeIf {modelInference.modelName==EyeAIApp.DEFAULT_DEPTH_MODEL_NAME},
                        metricFrame?.depthMeters?.copyOf(),modelInference.inputDim.width,modelInference.inputDim.height,
                        modelInference.modelName.takeIf {it==EyeAIApp.DEFAULT_DEPTH_MODEL_NAME},metricFrame?.let {"REL2ABS_${it.rel2absMode.name}_VISUAL"},metricFrame)
                    channelStatus(frame,"depth","AVAILABLE")
					runtime.describeChangedScene(frame.bitmap, frame.stamp)
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
						val objectInferenceMode =
							runtime.settings.maxObjectDetectionFrameRate?.let { gate.mode.name }
								?: "UNTHROTTLED"

						"Metric Depth model: ${modelInference.modelName}\n" + "Camera resolution: $inputResolution -> Depth model input: $modelInput\n" + "Inference mode: $objectInferenceMode\n\n" + "${uniffi.NativeLib.formattedDepthFrame()}\n" + "$formattedSourceFrame\n" + if (runtime.settings.enableObjectDetection) {
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
							debugInputBitmap = if (runtime.settings.showDebugInputBitmap) {
								frame.bitmap.copy(Bitmap.Config.ARGB_8888, false)
							} else null,
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
				worldState.analyzer.invalidateDepth(frame.stamp, Availability.INVALID)
                channelStatus(frame,"depth","INVALID")
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
		var reportedMode = gate.mode
		while (currentCoroutineContext().isActive) {
			observedSequence = frameAvailable.first { it > observedSequence }
			val frame = queuedFrame(contextObjects) ?: continue
			try {
				refreshMotionMonitoring()
				if (!runtime.settings.enableObjectDetection) {
                    channelStatus(frame,"objects","UNSUPPORTED")
					worldState.analyzer.invalidateObjects(frame.stamp, Availability.UNSUPPORTED)
					AIModelData.rel2AbsFrameCache.clearDetections()
					continue
				}

				val maxFrameRate = runtime.settings.maxObjectDetectionFrameRate
				if (maxFrameRate != null) {
					val now = AnalysisClock.nowNanos()
					gate.updateObjectDetectionBudget(maxFrameRate.toDouble(), now)
                    if(frame.capture!=null||contextNarrator.state.value.collecting)gate.onVisualSample(1.0,now)
					val visualScore = scene.sample(frame.bitmap, frame.rotationDegrees, now)?.let { sample ->
						if (!sample.baselineFrame) gate.onVisualSample(
							sample.score,
							sample.sampledAtNanos
						)
						sample.score.takeUnless { sample.baselineFrame }
					}
					val motionScore = motionLifecycle.score()
					val decision = gate.tryAcquire(
						phoneMotionScore = motionScore,
						nowNanos = now,
					)
					if (decision.mode != reportedMode) {
						Log.i(
							EyeAIApp.APP_LOG_TAG,
							"Object detection mode: $reportedMode -> ${decision.mode} " +
								"(visual=$visualScore, motion=$motionScore)",
						)
						reportedMode = decision.mode
					}
                    if (!decision.admitted) {
                        delay(decision.retryAfterNanos.nanoseconds)
                        if(frame.capture==null)continue
                        if(!gate.tryAcquire(motionLifecycle.score(),AnalysisClock.nowNanos()).admitted) {
                            channelStatus(frame,"objects","RATE_LIMITED")
                            continue
                        }
                    }
				}
				uniffi.NativeLib.newObjectFrame()
				contextNarrator.recorder.event("YOLO_MODEL_START",frame=frame.stamp)
                val objects = runtime.runObjectInference(frame.bitmap)
                contextNarrator.recorder.event("YOLO_MODEL_END",frame=frame.stamp)
				val safeObjects = objects ?: emptyArray()
				AIModelData.detectedObjects.set(safeObjects)
				val detectionFrame = DetectionFrame(
					detections = safeObjects,
					sourceTimestampNanos = frame.timestampNanos,
					sourceWidth = frame.width,
					sourceHeight = frame.height,
					rotationDegrees = frame.rotationDegrees,
					objectContextFeatures = Rel2AbsContextFeatures.objectFeatures(safeObjects),
				)
				AIModelData.rel2AbsFrameCache.publishDetections(detectionFrame)
                if(frame.capture!=null&&objects!=null)contextNarrator.capturedObjects(frame.stamp,detectionFrame)
                if(objects==null)channelStatus(frame,"objects","INVALID")else channelStatus(frame,"objects","AVAILABLE")
                contextNarrator.recorder.event("YOLO",frame=frame.stamp) { buildJsonArray {for(o in safeObjects)add(buildJsonObject {
                    put("track",o.trackingId)
put("class",o.cls)
put("name",o.clsName)
                    put("confidence",if(o.cnf.isFinite())JsonPrimitive(o.cnf)else JsonNull)
                    put("box",buildJsonArray {for(value in listOf(o.x1,o.y1,o.x2,o.y2))add(if(value.isFinite())JsonPrimitive(value)else JsonNull)})
                })} }
				if (objects == null) worldState.analyzer.invalidateObjects(frame.stamp)
				else if (runtime.settings.enableObjectDetection) {
                    if (contextNarrator.observing) worldState.objects(frame.stamp, detectionFrame)
                }
				else worldState.analyzer.invalidateObjects(frame.stamp, Availability.UNSUPPORTED)
				if (objects != null && runtime.settings.enableObjectDetection) {
					runtime.describeChangedScene(frame.bitmap, frame.stamp)
				}
				lastObjectOutputAtNanos.set(System.nanoTime())
				onUpdate(
					FrameAnalysisUpdate(
						detectedObjects = objects,
						frameSize = Size(frame.width, frame.height),
					)
				)
			} catch (cancelled: kotlinx.coroutines.CancellationException) {
				throw cancelled
			} catch (error: Throwable) {
				worldState.analyzer.invalidateObjects(frame.stamp, Availability.INVALID)
                channelStatus(frame,"objects","INVALID")
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
			val frame = queuedFrame(contextSegmentation) ?: continue
			try {
				if (!runtime.settings.enableSegmentation) {
                    channelStatus(frame,"segmentation","UNSUPPORTED")
					worldState.analyzer.invalidateSegmentation(frame.stamp, Availability.UNSUPPORTED)
					AIModelData.rel2AbsFrameCache.clearSegmentationContexts()
					continue
				}
				val inferenceDuration = measureTime {
					uniffi.NativeLib.newSegmentationFrame()
					contextNarrator.recorder.event("SEGMENTATION_MODEL_START",frame=frame.stamp)
                    val output = runtime.runSegmentationInference(frame.bitmap)
                    contextNarrator.recorder.event("SEGMENTATION_MODEL_END",frame=frame.stamp)
					if (output == null) {
                        channelStatus(frame,"segmentation","INVALID")
						worldState.analyzer.invalidateSegmentation(frame.stamp)
						continue
					}
					AIModelData.segmentationOutput.set(output.prediction)
					val segmentationContext = Rel2AbsContextFeatures.segmentationFeatureContext(
						output.prediction,
						output.inputDim.width,
						output.inputDim.height,
					)
					val contextFrame = SegmentationContextFrame(
						segmentationFeatures = Rel2AbsContextFeatures.segmentationFeatures(output.prediction),
						sourceTimestampNanos = frame.timestampNanos,
						sourceWidth = frame.width,
						sourceHeight = frame.height,
						rotationDegrees = frame.rotationDegrees,
						globalAreaFractions = segmentationContext.globalAreaFractions,
						gridAreaFractions = segmentationContext.gridAreaFractions,
						available = segmentationContext.available,
					)
					AIModelData.rel2AbsFrameCache.publishSegmentationContext(contextFrame)
                    if(frame.capture!=null)contextNarrator.capturedSegmentation(frame.stamp,contextFrame)
                    if(contextNarrator.recorder.state.value.active)contextNarrator.recorder.ints(frame.stamp,"segmentation-mask",output.inputDim.width,output.inputDim.height,
                        IntArray(output.prediction.intBuffer.capacity()) {output.prediction.intBuffer.get(it)})
                    channelStatus(frame,"segmentation",if(contextFrame.available)"AVAILABLE" else "INVALID")
					if (runtime.settings.enableSegmentation) {
                        if (contextNarrator.observing) worldState.segmentation(frame.stamp, contextFrame)
                    }
					else worldState.analyzer.invalidateSegmentation(frame.stamp, Availability.UNSUPPORTED)
					val colorMappedImage = if (runtime.settings.showSegmentationOverlay) {
                        NativeLib.segmentationColormap(output.prediction.asUniffiWrapper(), output.inputDim, output.classColors)
                    } else null
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
				worldState.analyzer.invalidateSegmentation(frame.stamp, Availability.INVALID)
                channelStatus(frame,"segmentation","INVALID")
				AIModelData.segmentationOutput.set(null)
				Log.e(EyeAIApp.APP_LOG_TAG, "Segmentation frame processing failed", error)
			} finally {
				frame.release()
			}
		}
	}

	private fun retainLatestFrame(): AcceptedFrame? {
		val frame = latestFrame.get() ?: return null
		return frame.takeIf { it.frame.tryRetain() }
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
		worldState.analyzer.reset()
		AIModelData.detectedObjects.set(emptyArray())
		AIModelData.depthEstimationData.set(null)
		AIModelData.segmentationOutput.set(null)
		AIModelData.rel2AbsFrameCache.clear()
		lastDepthOutputAtNanos.set(0L)
		lastObjectOutputAtNanos.set(0L)
	}

	// Gate and context observation share one motion sensor lifecycle.
	private fun refreshMotionMonitoring() {
		val operationActive = synchronized(stateLock) { startedValue && !shutdownValue }
		motionLifecycle.update(
			operationActive = operationActive,
			objectDetectionEnabled = runtime.settings.enableObjectDetection,
			limiterEnabled = runtime.settings.maxObjectDetectionFrameRate != null,
            narratorEnabled = contextNarrator.state.value.enabled || contextNarrator.recorder.state.value.active,
			profilingEnabled = runtime.settings.showProfilingInfo && runtime.settings.maxObjectDetectionFrameRate != null,
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
