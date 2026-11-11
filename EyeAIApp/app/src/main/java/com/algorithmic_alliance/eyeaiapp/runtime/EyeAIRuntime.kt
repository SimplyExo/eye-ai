package com.algorithmic_alliance.eyeaiapp.runtime

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.util.Log
import android.util.Size
import androidx.annotation.RequiresApi
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.Settings
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudio
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudioResumeController
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudioResumeOutcome
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.camera.FrameAnalysisUpdate
import com.algorithmic_alliance.eyeaiapp.camera.FrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.worldstate.Availability
import com.algorithmic_alliance.eyeaiapp.connectivity.EyeAIVision
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModel
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModelInfo
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.EyeAIState
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.GenericCancellation
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateMachine
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.VoskRestartPolicy
import com.algorithmic_alliance.eyeaiapp.media.MediaPlayer
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModel
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModelInfo
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModel
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModelInfo
import com.algorithmic_alliance.eyeaiapp.ocr.GoogleOCR
import com.algorithmic_alliance.eyeaiapp.rel2abs.Rel2AbsRunner
import com.algorithmic_alliance.eyeaiapp.rel2abs.V6NeuralGateRunner
import com.algorithmic_alliance.eyeaiapp.segmentation.SegmentationModel
import com.algorithmic_alliance.eyeaiapp.segmentation.SegmentationModelInfo
import com.algorithmic_alliance.eyeaiapp.speech_recognition.VoskModel
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import com.algorithmic_alliance.eyeaiapp.vibrate
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import uniffi.NativeLib.UniffiDetectedObject
import java.lang.ref.WeakReference
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import androidx.core.net.toUri
import kotlin.time.Duration.Companion.milliseconds

/** Output of a depth inference while the model read lock is held. */
data class DepthInferenceResult(
	val prediction: NativeLib.NativeFloatBuffer,
	val rawRelativeDepth: NativeLib.NativeFloatBuffer,
	val inputDim: Size,
	val modelName: String,
)

/** Output of a depth inference while the model read lock is held. */
data class SegmentationInferenceResult(
	val prediction: NativeLib.NativeIntBuffer,
	val inputDim: Size,
	val classColors: List<List<Int>>,
	val modelName: String,
)

/**
 * The process-wide EyeAI runtime. It is created exactly once by
 * [EyeAIApp], while [EyeAIRuntimeService] owns the active operation lifecycle.
 * No field in this class references an Activity, View, or Compose lifecycle.
 */
class EyeAIRuntime internal constructor(
	private val app: EyeAIApp,
) {
	init {
		Log.e(EyeAIApp.APP_LOG_TAG, "!!! EyeAIRuntime INITIALIZED !!!")
	}
	private val context: Context = app.applicationContext
	private val lifecycleGate = RuntimeLifecycleGate()
	private val stateLock = Any()
	private val modelLock = ReentrantReadWriteLock()

	private val _state = MutableStateFlow(EyeAIRuntimeState())
	val state: StateFlow<EyeAIRuntimeState> = _state.asStateFlow()

	private val runtimeJob = SupervisorJob()
	private val runtimeScope = CoroutineScope(runtimeJob + Dispatchers.Default)
	private val speechThreadExecutor = Executors.newSingleThreadExecutor()
	private val speechDispatcher = speechThreadExecutor.asCoroutineDispatcher()
	private val speechScope = CoroutineScope(runtimeJob + speechDispatcher)
	private val modelExecutor = Executors.newSingleThreadExecutor()
	private val modelScope = CoroutineScope(runtimeJob + modelExecutor.asCoroutineDispatcher())
	private val modelLoadRequested = AtomicBoolean(false)

	private var metricDepthModelValue: MetricDepthModel? = null
	private var rel2AbsRunnerValue: Rel2AbsRunner? = null
	private var rel2AbsNeuralGateRunnerValue: V6NeuralGateRunner? = null
	private var textToSpeechInstanceValue: TextToSpeechInstance? = null
	private var speechCallbacksInstalled = false
	private var lastFinalResultMillis = 0L
	private var voskStarting = AtomicBoolean(false)
	private var currentState = EyeAIState.IDLE
	private var lastDialogContextValue: String? = null
	private var runtimeClosed = false
	private var serviceOwner = WeakReference<LifecycleOwner>(null)
	private var mediaPlayerValue: MediaPlayer? = null
	private var eyeAIVisionValue: EyeAIVision? = null
	private var bitmapFlowValue: MutableSharedFlow<Bitmap>? = null
	private var analysisHealthJob: kotlinx.coroutines.Job? = null
	private var lastCameraRestartAtNanos = 0L

	val spatialAudioResumeController = SpatialAudioResumeController(
		scope = runtimeScope,
		pauseSpatialAudio = ::pauseSpatialAudio,
		restoreSpatialAudio = ::restoreSpatialAudioFromSettings,
		awaitTtsSilence = {
			textToSpeechInstance.awaitSilence(quietMs = 500L, maxWaitMs = 30_000L)
		},
		isListening = { voskUserStart.get() },
		onOutcome = { trigger, outcome ->
			when (outcome) {
				SpatialAudioResumeOutcome.RESTORED -> Unit
				SpatialAudioResumeOutcome.TTS_SILENCE_TIMEOUT -> Log.w(
					EyeAIApp.APP_LOG_TAG,
					"[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=SKIPPED reason=TTS_SILENCE_TIMEOUT",
				)

				SpatialAudioResumeOutcome.LISTENING_STATE_CHANGED -> Log.d(
					EyeAIApp.APP_LOG_TAG,
					"[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=SKIPPED reason=LISTENING_STATE_CHANGED",
				)
			}
		},
	)

	val voskUserStart = AtomicBoolean(false)
	val yoloModel = YoloModel(YoloModelInfo("yolo26n.tflite", "coco.names", 640))
	val segmentationModel =
		SegmentationModel(SegmentationModelInfo("yolo26n-sem.tflite", "yolo26n-sem.names.json", 256))
	val nlpModel = NLPModel(NLPModelInfo.findById(NLPModelInfo.DEFAULT_MODEL_ID))
	val ocrModel = GoogleOCR()
	val voskModel = VoskModel(context, "model-de")
	val frameAnalyzer = FrameAnalyzer(context, this, ::onFrameAnalysisUpdate)
	val worldStateAnalyzer get() = frameAnalyzer.worldState.analyzer
	val localSceneDescriptions = com.algorithmic_alliance.eyeaiapp.scenenarrator.LocalSceneDescriptions(context)
	@Volatile private var orientationSpeechId: Long? = null
    init {
        runtimeScope.launch {
            localSceneDescriptions.state.map {it.episodeId}.distinctUntilChanged().collect {
                if(textToSpeechInstanceValue?.stopIfOwnerPrefix("orientation:")==true) {
                    localSceneDescriptions.recorder.event("SPEECH_CANCELLED_NEW_CONTEXT")
                    orientationSpeechId=null
                    _state.update {state->state.copy(ttsSpeaking=false)}
                }
            }
        }
		runtimeScope.launch {
			localSceneDescriptions.state.map { if (it.enabled) it.announcementId else 0L }
				.distinctUntilChanged().collectLatest { id ->
					if (id == 0L) return@collectLatest
					val tts = textToSpeechInstance
					while (this@EyeAIRuntime.isActive && localSceneDescriptions.isCurrentAnnouncement(id) &&
						(voskUserStart.get() || tts.isSpeaking() || !tts.isReadyToSpeak() || !localSceneDescriptions.speechStartAllowed())) delay(150L)
					withContext(Dispatchers.Main.immediate) {
						if (this@EyeAIRuntime.isActive && localSceneDescriptions.isCurrentAnnouncement(id) &&
							!voskUserStart.get() && !tts.isSpeaking() && tts.isReadyToSpeak() && localSceneDescriptions.speechStartAllowed()) {
							orientationSpeechId = id
							speakSceneText(localSceneDescriptions.state.value.text) {
								if (orientationSpeechId == id) orientationSpeechId = null
							}
						}
					}
				}
		}
	}
	fun setLocalSceneDescriptionsEnabled(enabled: Boolean) {
		localSceneDescriptions.setEnabled(enabled && settings.isR4Mode && isActive)
		if (!localSceneDescriptions.state.value.enabled) {
			textToSpeechInstanceValue?.stopIfOwnerPrefix("orientation:")
			orientationSpeechId = null
			_state.update { it.copy(ttsSpeaking = textToSpeechInstanceValue?.isSpeaking() == true) }
		}
	}

	fun setLocalSceneR4ExperimentEnabled(enabled: Boolean) {
		localSceneDescriptions.setExperimentalR4Enabled(enabled)
		if (orientationSpeechId != null) {
			textToSpeechInstance.stopIfOwnerPrefix("orientation:")
			orientationSpeechId = null
			_state.update { it.copy(ttsSpeaking = false) }
		}
	}

    fun setContextRecording(enabled: Boolean) {
        val recorder=localSceneDescriptions.recorder
        if(!enabled) {recorder.stop()
        return}
        recorder.start()
        localSceneDescriptions.recordConfiguration()
        val depth=EyeAIApp.DEPTH_MODELS.firstOrNull {it.name==settings.depthModel}?.relativeDepthFileName
        val files=listOfNotNull(depth,yoloModel.info.tfliteFilename,yoloModel.info.namesFilename,segmentationModel.info.tfliteFilename,segmentationModel.info.namesJsonFilename)+
            listOf("context-policy.json","scene-narrator/r4-model.tflite","scene-narrator/deterministic-manifest.json","scene-narrator/evidence-contract.json","scene-narrator/r4-manifest.json","scene-narrator/r4-gate.json","scene-narrator/r4-experimental-policy.json")+
            context.assets.list("rel2abs").orEmpty().filter {it.endsWith(".json")||it.endsWith(".tflite")}.map {"rel2abs/$it"}
        recorder.assetMetadata(files)
    }

    internal fun observeContextFrame(bitmap: Bitmap,stamp: com.algorithmic_alliance.eyeaiapp.worldstate.FrameStamp,motion: Double?): com.algorithmic_alliance.eyeaiapp.context.ContextSnapshot? =
        if(isActive) {
            localSceneDescriptions.setBackendConfiguration("${settings.depthModel}|${settings.rel2AbsMode}|${settings.cameraHeightTenths}|${settings.enableObjectDetection}|${settings.enableSegmentation}")
            localSceneDescriptions.onCameraFrame(bitmap,stamp,{ worldStateAnalyzer.snapshot() },motion)
        } else null

	// Called synchronously while a model worker owns this bitmap.
	internal fun describeChangedScene(bitmap: Bitmap, stamp: com.algorithmic_alliance.eyeaiapp.worldstate.FrameStamp) {
		if (isActive) localSceneDescriptions.onFrame(bitmap, stamp) { worldStateAnalyzer.snapshot() }
	}

	fun speakLocalSceneDescription() {
		val text = localSceneDescriptions.state.value.text
		if (!isActive || !settings.isR4Mode || text.isBlank()) return
		if (voskUserStart.get()) stopListening("LOCAL_SCENE_DESCRIPTION")
		speakSceneText(text)
	}

	private fun speakSceneText(text: String, onComplete: (() -> Unit)? = null) {
        if (!isActive || !settings.isR4Mode) return
		pauseSpatialAudio()
		_state.update { it.copy(speechResponseText = if(settings.sceneDescriptionSpatialAudio)localSceneDescriptions.state.value.segments.joinToString(" "){it.text} else text, ttsSpeaking = true) }
		val id=localSceneDescriptions.state.value.announcementId
        val valid = { isActive && settings.isR4Mode && localSceneDescriptions.isCurrentAnnouncement(id) }
        val admitted=if(settings.sceneDescriptionSpatialAudio) textToSpeechInstance.speakSpatialOwned(
            localSceneDescriptions.state.value.segments,"orientation:$id",
            valid=valid,
            invalidated=localSceneDescriptions.state.filter { !valid() }.map { Unit },
            onStarted={localSceneDescriptions.speechStarted(id)},
            onComplete={localSceneDescriptions.speechFinished(id);onComplete?.invoke()},
            onCancelled={localSceneDescriptions.speechCancelled(id);onComplete?.invoke()}
        ) else textToSpeechInstance.speakOwned(text,"orientation:$id",onStarted={localSceneDescriptions.speechStarted(id)},onComplete={
            localSceneDescriptions.speechFinished(id)
            onComplete?.invoke()
        })
        if(!admitted) {orientationSpeechId=null
_state.update {it.copy(ttsSpeaking=textToSpeechInstance.isSpeaking())}
        return}
		spatialAudioResumeController.schedule("LOCAL_SCENE_DESCRIPTION")
	}
	val cameraManager = CameraManager(::onCameraStateChanged)
	val npuQnnDelegateDirectory: String = app.applicationInfo.nativeLibraryDir

	val settings: Settings
		get() = app.settings

	private val rel2AbsRunner: Rel2AbsRunner
		get() = synchronized(stateLock) {
			check(!runtimeClosed) { "EyeAI runtime is closed" }
			rel2AbsRunnerValue ?: Rel2AbsRunner.fromAssets(context).also {
				rel2AbsRunnerValue = it
			}
		}

	internal val rel2AbsNeuralGateRunner: V6NeuralGateRunner?
		get() = synchronized(stateLock) {
			if (!settings.rel2AbsMode.isNeuralGate || runtimeClosed) return@synchronized null
			rel2AbsNeuralGateRunnerValue ?: V6NeuralGateRunner.fromAssets(context).also {
				rel2AbsNeuralGateRunnerValue = it
			}
		}

	val textToSpeechInstance: TextToSpeechInstance
		get() = synchronized(stateLock) {
			check(!runtimeClosed) { "EyeAI runtime is closed" }
			textToSpeechInstanceValue ?: TextToSpeechInstance(
				context = context,
				onTTSFinishedSpeaking = ::onTtsFinishedSpeaking,
			).also { textToSpeechInstanceValue = it }
		}

	var lastDialogContext: String?
		get() = synchronized(stateLock) { lastDialogContextValue }
		private set(value) {
			synchronized(stateLock) { lastDialogContextValue = value }
		}

	val isActive: Boolean
		get() = lifecycleGate.isActive

	fun initializeModels() {
		if (!modelLoadRequested.compareAndSet(false, true)) return
		modelScope.launch {
			try {
				switchDepthModel(settings.depthModel)
				if (settings.enableObjectDetection) {
					yoloModel.create(context, npuQnnDelegateDirectory, settings.enableNpu)
				}
				if (settings.enableSegmentation) {
					segmentationModel.create(context, npuQnnDelegateDirectory, settings.enableNpu)
					app.aiData.segmentationClassImportances.set(segmentationModel.classImportances)
				}
				switchNlpModel(settings.nlpModel)
				if (settings.enableOCR) ocrModel.create()
			} catch (error: Throwable) {
				Log.e(EyeAIApp.APP_LOG_TAG, "Initial AI model loading failed", error)
			}
		}
	}

	/** Called by the Application after settings have been reloaded. */
	fun onSettingsChanged(oldSettings: Settings) {
		val newSettings = settings
        if (oldSettings.audioOutputMode != newSettings.audioOutputMode) {
            setLocalSceneDescriptionsEnabled(false)
            if (isActive) {
                localSceneDescriptions.setExperimentalR4Enabled(true)
                setLocalSceneDescriptionsEnabled(newSettings.isR4Mode)
            }
        }
        if(oldSettings.sceneDescriptionSpatialAudio != newSettings.sceneDescriptionSpatialAudio) {
            textToSpeechInstanceValue?.stopIfOwnerPrefix("orientation:")
            orientationSpeechId=null
        }
        if(oldSettings.sceneDescriptionDetail != newSettings.sceneDescriptionDetail) {
            localSceneDescriptions.setDetail(newSettings.sceneDescriptionDetail)
            if(orientationSpeechId != null) {
                textToSpeechInstance.stopIfOwnerPrefix("orientation:")
                orientationSpeechId=null
                _state.update {it.copy(ttsSpeaking=false)}
            }
        }
		if (oldSettings.rel2AbsMode != newSettings.rel2AbsMode) {
			// Never mix a depth buffer produced with one frozen REL2ABS mode with
			// detections paired after the user selected the other one.
			app.aiData.rel2AbsFrameCache.clear()
		}
		if (oldSettings.effectiveDepthAudioPlayback != newSettings.effectiveDepthAudioPlayback) {
			uniffi.NativeLib.setDepthAudioPaused(!newSettings.effectiveDepthAudioPlayback)
		}
		if (oldSettings.effectiveObjectAudioPlayback != newSettings.effectiveObjectAudioPlayback) {
			uniffi.NativeLib.setObjectAudioPaused(!newSettings.effectiveObjectAudioPlayback)
		}
		if (oldSettings.depthAudioFrequency != newSettings.depthAudioFrequency || oldSettings.depthAudioClickIncidence != newSettings.depthAudioClickIncidence) {
			uniffi.NativeLib.setAudioSettings(
				newSettings.depthAudioFrequency.toFloat(),
				newSettings.depthAudioClickIncidence,
			)
		}

		if (!newSettings.enableObjectDetection) worldStateAnalyzer.invalidateObjects(status = Availability.UNSUPPORTED)
		if (!newSettings.enableSegmentation) worldStateAnalyzer.invalidateSegmentation(status = Availability.UNSUPPORTED)
		modelScope.launch {
			try {
				if (oldSettings.nlpModel != newSettings.nlpModel) {
					switchNlpModel(newSettings.nlpModel)
				}
				if (oldSettings.depthModel != newSettings.depthModel || oldSettings.enableNpu != newSettings.enableNpu) {
					switchDepthModel(newSettings.depthModel)
				}
				if (newSettings.enableObjectDetection && (!oldSettings.enableObjectDetection || oldSettings.enableNpu != newSettings.enableNpu)) {
					yoloModel.create(context, npuQnnDelegateDirectory, newSettings.enableNpu)
				}
				if (newSettings.enableSegmentation && (!oldSettings.enableSegmentation || oldSettings.enableNpu != newSettings.enableNpu)) {
					segmentationModel.create(
						context, npuQnnDelegateDirectory, newSettings.enableNpu
					)
					app.aiData.segmentationClassImportances.set(segmentationModel.classImportances)
				}
				if (newSettings.enableOCR && !oldSettings.enableOCR) ocrModel.create()
				if (!newSettings.enableOCR && oldSettings.enableOCR) ocrModel.close()
				if (isActive && (oldSettings.objectAudioPlaybackLanguage != newSettings.objectAudioPlaybackLanguage ||
                    oldSettings.audioOutputMode != newSettings.audioOutputMode)) {
					SpatialAudio.setup(context)
                    if (voskUserStart.get() || textToSpeechInstance.isSpeaking()) pauseSpatialAudio()
                }
				if (!newSettings.enableSpeechRecognition) closeSpeechService()
				if (isActive && oldSettings.enableSpeechRecognition != newSettings.enableSpeechRecognition) {
					// Drop the microphone use before removing its FGS type;
					// add the type before starting a newly enabled listener.
					(serviceOwner.get() as? EyeAIRuntimeService)?.refreshForegroundTypes()
				}
				if (newSettings.enableSpeechRecognition && isActive) initSpeechService()
				if (oldSettings.inputSource != newSettings.inputSource && isActive) {
					// Source changes can also change the FGS type and the
					// wake-lock requirement. End this operation cleanly and
					// let the visible UI start a new service with the new
					// source configuration. Models remain runtime-owned and
					// are not reloaded by this transition.
					EyeAIRuntimeService.stop(context)
					return@launch
				}
				if (oldSettings.mediaSource != newSettings.mediaSource &&
					newSettings.inputSource == context.getString(R.string.input_is_media) && isActive
				) {
					// The media file changed while the media source is active.
					// Stop so the visible UI restarts the service with the new
					// media URI instead of continuing to play the old file.
					EyeAIRuntimeService.stop(context)
					return@launch
				}
			} catch (error: Throwable) {
				Log.e(EyeAIApp.APP_LOG_TAG, "Settings-triggered runtime update failed", error)
			}
		}
	}

	/** Starts the active local source under the foreground service lifecycle. */
	@RequiresApi(Build.VERSION_CODES.P)
	fun start(owner: LifecycleOwner) {
		Log.e(EyeAIApp.APP_LOG_TAG, "!!! EyeAIRuntime.start() called !!!")
		synchronized(stateLock) {
			check(!runtimeClosed) { "EyeAI runtime is closed" }
		}
		if (!lifecycleGate.start()) return

		try {
			initializeModels()
			textToSpeechInstance
			// Only the R4 output mode enables automatic scene orientation.
			localSceneDescriptions.setDetail(settings.sceneDescriptionDetail)
			localSceneDescriptions.setExperimentalR4Enabled(true)
			setLocalSceneDescriptionsEnabled(settings.isR4Mode)
			frameAnalyzer.start()
			SpatialAudio.setup(context)
			SpatialAudio.start()
			serviceOwner = WeakReference(owner)
			startVideoSource(owner)
			startAnalysisHealthMonitor()
			if (settings.enableSpeechRecognition && hasRecordAudioPermission()) {
				initSpeechService()
			}
			if (voskUserStart.get() || textToSpeechInstance.isSpeaking()) {
				pauseSpatialAudio()
			} else {
				restoreSpatialAudioFromSettings("RUNTIME_START")
			}
			_state.update { it.copy(operationActive = true, lastError = null) }
		} catch (error: Throwable) {
			Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI runtime start failed", error)
			stopAfterStartFailure()
			throw error
		}
	}

	/** Attaches/detaches only the optional UI preview surface. */
	fun attachPreview(previewView: PreviewView?) = cameraManager.attachPreview(previewView)

	fun detachPreview(previewView: PreviewView? = null) = cameraManager.detachPreview(previewView)

	/** Explicit user stop. Models stay cached; active input resources do not. */
	@RequiresApi(Build.VERSION_CODES.P)
	fun stopOperation() {
		if (!lifecycleGate.stop()) return
		localSceneDescriptions.stop()
        localSceneDescriptions.recorder.stop()
		cleanupStep("spatial-audio resume controller") {
			spatialAudioResumeController.cancel()
		}
		cleanupStep("analysis health monitor") {
			analysisHealthJob?.cancel()
			analysisHealthJob = null
		}
		cleanupStep("frame analyzer") { frameAnalyzer.stop() }
		cleanupStep("video source") { stopVideoSource() }
		voskUserStart.set(false)
		cleanupStep("Vosk listener") {
			voskModel.stopListening()
			eyeAIVisionValue?.stopAudioStream()
		}
		cleanupStep("TTS") { synchronized(stateLock) { textToSpeechInstanceValue }?.stop() }
		orientationSpeechId = null
		cleanupStep("spatial-audio engine") { SpatialAudio.stop() }
		cleanupStep("native spatial-audio pause") { pauseSpatialAudio() }
		_state.update {
			it.copy(
				operationActive = false,
				cameraActive = false,
				voskListening = false,
				ttsSpeaking = false,
			)
		}
	}

	fun toggleListening() {
		if (textToSpeechInstance.isSpeaking()) {
			textToSpeechInstance.stop()
			orientationSpeechId = null
			_state.update { it.copy(ttsSpeaking = false, speechResponseText = "") }
			return
		}
		if (voskUserStart.get()) stopListening()
		else startListening("USER_BUTTON")
	}

	fun initSpeechService() {
		if (!settings.enableSpeechRecognition || !hasRecordAudioPermission()) {
			publishVoskStatus(false)
			return
		}
		synchronized(stateLock) {
			if (speechCallbacksInstalled) return
			speechCallbacksInstalled = true
		}
		voskModel.initService(
			onPartialResult = ::onPartialSpeechRecognitionResult,
			onFinalResult = ::onFinalSpeechRecognitionResult,
			onModelLoaded = ::onSpeechRecognitionLoaded,
			onUpdateVoskUIStatus = ::publishVoskStatus,
		)
	}

	fun closeSpeechService() {
		voskUserStart.set(false)
		voskModel.stopListening()
		eyeAIVisionValue?.stopAudioStream()
		voskModel.closeService()
		publishVoskStatus(false)
		synchronized(stateLock) { speechCallbacksInstalled = false }
	}

	private fun startListening(trigger: String) {
		if (!settings.enableSpeechRecognition || !hasRecordAudioPermission()) {
			publishVoskStatus(false)
			return
		}
		if (!isActive) return
		if (voskUserStart.getAndSet(true)) return
		spatialAudioResumeController.cancel()
		pauseSpatialAudio()
		initSpeechService()
		lastFinalResultMillis = 0L

		val vision = eyeAIVisionValue
		if (vision != null) {
			speechScope.launch(Dispatchers.IO) {
				val inputStream = vision.startAudioStream()
				withContext(Dispatchers.Main) {
					if (!voskUserStart.get()) {
						vision.stopAudioStream()
						return@withContext
					}
					if (inputStream != null) {
						voskModel.startListening(inputStream)
					} else {
						Log.w(
							EyeAIApp.APP_LOG_TAG,
							"[EyeAIRuntime] Network microphone on port 4040 unavailable; falling back to local mic"
						)
						voskModel.startListening()
					}
					if (!voskModel.isListening()) {
						voskUserStart.set(false)
						vision.stopAudioStream()
						restoreSpatialAudioFromSettings("${trigger}_MODEL_NOT_READY")
						return@withContext
					}
					Log.i(
						EyeAIApp.APP_LOG_TAG,
						"[DecisionTrace][Vosk][START] trigger=$trigger outcome=LISTENING (network_mic=${inputStream != null})",
					)
					publishVoskStatus(true)
				}
			}
		} else {
			voskModel.startListening()
			if (!voskModel.isListening()) {
				voskUserStart.set(false)
				restoreSpatialAudioFromSettings("${trigger}_MODEL_NOT_READY")
				return
			}
			Log.i(
				EyeAIApp.APP_LOG_TAG,
				"[DecisionTrace][Vosk][START] trigger=$trigger outcome=LISTENING",
			)
			publishVoskStatus(true)
		}
	}

	private fun stopListening(
		trigger: String = "USER_BUTTON",
		restoreAfterTts: Boolean = false,
	) {
		if (!voskUserStart.getAndSet(false)) return
		voskModel.stopListening()
		eyeAIVisionValue?.stopAudioStream()
		if (restoreAfterTts) spatialAudioResumeController.schedule(trigger)
		else {
			spatialAudioResumeController.cancel()
			restoreSpatialAudioFromSettings(trigger)
		}
		publishVoskStatus(false)
		Log.i(
			EyeAIApp.APP_LOG_TAG,
			"[DecisionTrace][Vosk][STOP] trigger=$trigger outcome=STOPPED",
		)
	}

	private fun onPartialSpeechRecognitionResult(partial: String) {
		_state.update { it.copy(speechRecognitionPartialResultText = partial) }
		if (partial.isNotEmpty()) {
			Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIRuntime] Partial speech recognized: '$partial'")
		}
	}

	private fun onSpeechRecognitionLoaded() {
		updateVoskStatusText()
	}

	private fun onFinalSpeechRecognitionResult(final: String) {
		if (final.isEmpty()) return
		val now = System.currentTimeMillis()
		if (now - lastFinalResultMillis <= 1_000L) {
			Log.w(
				EyeAIApp.APP_LOG_TAG,
				"[EyeAIRuntime] Ignored duplicate speech recognition result within 1s: '$final'"
			)
			return
		}
		lastFinalResultMillis = now
		Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIRuntime] Final speech recognized: '$final'")
		val receiveTs = System.nanoTime()

		_state.update {
			it.copy(
				speechRecognitionFinalResultText = final,
			)
		}

		voskModel.stopListening(playSound = false)
		eyeAIVisionValue?.stopAudioStream()
		vibrate(context, 100)
		speechScope.launch {
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"[DecisionTrace][StateMachine][DISPATCH] state=$currentState; " + "latencySinceVosk=${(System.nanoTime() - receiveTs) / 1_000_000}ms",
			)
			processSpeech(final)
		}
	}

	private suspend fun processSpeech(final: String) {
		withContext(speechDispatcher) {
			_state.update { it.copy(ttsSpeaking = true) }
			val stateMachine = StateMachine(
				eyeAIApp = app,
				textToSpeechInstance = textToSpeechInstance,
				lastDialogContext = lastDialogContext,
				setSpeechResponseText = { response ->
					_state.update { it.copy(speechResponseText = response) }
				},
				frameAnalyzer = frameAnalyzer,
			)
			val cancellationResponse = GenericCancellation.responseFor(final)
			val update = if (cancellationResponse != null) {
				stateMachine.handleCancellation()
			} else {
				when (currentState) {
					EyeAIState.IDLE -> stateMachine.handleIdle(final)
					EyeAIState.SETTINGS_MENU -> stateMachine.handleSettingsMenu(final)
					EyeAIState.SETTINGS_CHOICE -> stateMachine.handleSettingsChoice(final)
					EyeAIState.SETTINGS_ACTION -> stateMachine.handleSettingsAction(final)
					EyeAIState.SETTINGS_EXTERNAL_CONFIRMATION -> stateMachine.handleSettingsExternalConfirmation(
						final
					)
				}
			}
			if (update.voskRestartPolicy == VoskRestartPolicy.REQUIRE_MANUAL_RESTART) {
				stopListening("SETTINGS_APPLIED", restoreAfterTts = true)
			}
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"[DecisionTrace][StateMachine][TRANSITION] $currentState -> ${update.newState}; " + "voskRestartPolicy=${update.voskRestartPolicy}",
			)
			currentState = update.newState
			lastDialogContext = update.newJson
		}
	}

	private fun onTtsFinishedSpeaking() {
		_state.update { it.copy(ttsSpeaking = false) }
		if (!isActive || !voskUserStart.get()) return
		if (!voskStarting.compareAndSet(false, true)) return
		runtimeScope.launch {
			try {
				if (isActive && voskUserStart.get()) {
					lastFinalResultMillis = 0L
					val vision = eyeAIVisionValue
					if (vision != null) {
						val inputStream = withContext(Dispatchers.IO) { vision.startAudioStream() }
						if (isActive && voskUserStart.get()) {
							if (inputStream != null) {
								voskModel.startListening(inputStream, playSound = true)
							} else {
								voskModel.startListening(playSound = true)
							}
						} else {
							withContext(Dispatchers.IO) { vision.stopAudioStream() }
						}
					} else {
						voskModel.startListening(playSound = true)
					}
					publishVoskStatus(voskModel.isListening())
				}
			} catch (error: Throwable) {
				Log.e(EyeAIApp.APP_LOG_TAG, "Vosk restart after TTS failed", error)
			} finally {
				voskStarting.set(false)
			}
		}
	}

	fun updateVoskStatusText() {
		val text = when {
			!hasRecordAudioPermission() -> context.getString(R.string.vosk_card_missing_permission)
			!settings.enableSpeechRecognition -> context.getString(R.string.vosk_card_disabled)
			voskUserStart.get() -> context.getString(R.string.vosk_card_listening)
			else -> context.getString(R.string.vosk_card_ready)
		}
		_state.update { it.copy(speechRecognitionFinalResultText = text) }
	}

	private fun publishVoskStatus(status: Boolean) {
		updateVoskStatusText()
		_state.update { it.copy(voskListening = status) }
	}

	private fun onFrameAnalysisUpdate(update: FrameAnalysisUpdate) {
		_state.update { it.withAnalysis(update) }
	}

	private fun onCameraStateChanged(running: Boolean, error: Throwable?) {
		_state.update {
			it.copy(
				cameraActive = running,
				lastError = error?.message,
			)
		}
	}

	private fun startVideoSource(owner: LifecycleOwner) {
		val source = settings.inputSource
		Log.e(EyeAIApp.APP_LOG_TAG, "!!! EyeAIRuntime starting video source: $source !!!")
		when (source) {
			context.getString(R.string.input_is_camera) -> {
				Log.e(EyeAIApp.APP_LOG_TAG, "!!! Using Camera source !!!")
				cameraManager.start(
					context = context,
					owner = owner,
					preferredImageSize = EyeAIApp.PREFERRED_CAMERA_RESOLUTION,
					cameraPreviewView = null,
					frameAnalyzer = frameAnalyzer,
				)
			}

			context.getString(R.string.input_is_media) -> {
				Log.e(EyeAIApp.APP_LOG_TAG, "!!! Using Media source !!!")
				val mediaSource = settings.mediaSource
				if (mediaSource.isNullOrEmpty()) {
					_state.update { it.copy(lastError = "Keine Media-Quelle ausgewählt") }
				} else if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
					_state.update {
						it.copy(lastError = "Media-Eingabe benötigt Android 9 oder neuer")
					}
				} else {
					startMediaSource(mediaSource.toUri())
				}
			}

			context.getString(R.string.input_is_eyeaivision) -> {
				Log.e(EyeAIApp.APP_LOG_TAG, "!!! Using EyeAI-Vision source !!!")
				if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
					_state.update {
						it.copy(lastError = "EyeAI-Vision-Eingabe benötigt Android 9 oder neuer")
					}
				} else {
					startEyeAIVisionSource()
				}
			}

			else -> {
				Log.e(EyeAIApp.APP_LOG_TAG, "!!! Unknown input source: $source !!!")
				_state.update { it.copy(lastError = "Unbekannte Eingabequelle: $source") }
			}
		}
	}

	//Detects when the frame source stops delivering new frames
	//If no valid frame arrives for too long, old detection results are cleared
	//so SpatialAudio and object detection do not continue using stale data.
	//For CameraX, the camera is restarted once automatically.
	//External sources are only reported and cleared here, reconnecting them is
	//handled elsewhere.

	private fun startAnalysisHealthMonitor() {
		analysisHealthJob?.cancel()
		lastCameraRestartAtNanos = 0L
		val monitorStartedAtNanos = System.nanoTime()
		analysisHealthJob = runtimeScope.launch {
			var stallReported = false
			while (isActive && lifecycleGate.isActive) {
				delay(2_000L.milliseconds)
				val now = System.nanoTime()
				val health = frameAnalyzer.healthSnapshot()
				val ageNanos = if (health.lastAcceptedFrameAtNanos == 0L) {
					now - monitorStartedAtNanos
				} else {
					now - health.lastAcceptedFrameAtNanos
				}
				val stillStarting = health.lastAcceptedFrameAtNanos == 0L &&
					(now - monitorStartedAtNanos) < INPUT_START_GRACE_NANOS
				if (stillStarting) continue

				if (ageNanos > INPUT_STALE_TIMEOUT_NANOS) {
					if (!stallReported) {
						stallReported = true
						frameAnalyzer.clearModelOutputs()
						pauseSpatialAudio()
						Log.e(
							EyeAIApp.APP_LOG_TAG,
							"Analysis input stalled for ${ageNanos / 1_000_000} ms; clearing model outputs",
						)
						_state.update {
							it.copy(lastError = it.lastError ?: "Kein neues Kamerabild empfangen")
						}
					}

					if (settings.inputSource == context.getString(R.string.input_is_camera) &&
						now - lastCameraRestartAtNanos > CAMERA_RESTART_COOLDOWN_NANOS
					) {
						lastCameraRestartAtNanos = now
						val owner = serviceOwner.get()
						if (owner != null) {
							Log.w(EyeAIApp.APP_LOG_TAG, "Restarting stalled CameraX analysis source")
							cameraManager.stop()
							cameraManager.start(
								context = context,
								owner = owner,
								preferredImageSize = EyeAIApp.PREFERRED_CAMERA_RESOLUTION,
								cameraPreviewView = null,
								frameAnalyzer = frameAnalyzer,
							)
						}
					}
				} else if (stallReported) {
					stallReported = false
					Log.i(EyeAIApp.APP_LOG_TAG, "Analysis input recovered; restoring configured spatial audio")
					_state.update { it.copy(lastError = null) }
					if (!voskUserStart.get() && !textToSpeechInstance.isSpeaking()) {
						restoreSpatialAudioFromSettings("ANALYSIS_RECOVERED")
					}
				}
			}
		}
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun startMediaSource(uri: Uri) {
		mediaPlayerValue = MediaPlayer(
			context = context,
			uri = uri,
			updateTargetImageView = { bitmap ->
				_state.update { it.copy(mediaPreviewBitmap = bitmap) }
			},
			onFrame = { bitmap -> frameAnalyzer.submitBitmap(bitmap) },
		)
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun startEyeAIVisionSource() {
		val flow = MutableSharedFlow<Bitmap>(replay = 1, extraBufferCapacity = 1)
		val streamFailed = AtomicBoolean(false)
		_state.update { it.copy(mediaPreviewBitmap = null) }
		bitmapFlowValue = flow

		// Support for WebRTC (WHEP)
		val visionIp = "192.168.4.1"

		eyeAIVisionValue = EyeAIVision(
			app,
			ip = visionIp,
			onSingleClick = { startListening("EYEAIVISION_BUTTON") },
			onDoubleClick = { stopListening("EYEAIVISION_BUTTON") },
			onConnectingSocket = {},
			onSocketConnectionEstablished = {},
			onSocketFailed = { error ->
				Log.e(EyeAIApp.APP_LOG_TAG, "!!! Socket failed: ${error.message} !!!")
				_state.update { it.copy(lastError = error.message) }
			},
			onStreamError = { message ->
				streamFailed.set(true)
				_state.update { it.copy(lastError = message, mediaPreviewBitmap = null) }
			},
			onWebrtcFrame = { bitmap: Bitmap ->
				if (streamFailed.getAndSet(false)) {
					_state.update { it.copy(lastError = null) }
				}
				val success = flow.tryEmit(bitmap)
				Log.v(EyeAIApp.APP_LOG_TAG, "WebRTC frame eingelesen")
				if (!success) {
					Log.v(EyeAIApp.APP_LOG_TAG, "WebRTC frame dropped (flow full)")
				}
			}
		)

		mediaPlayerValue = MediaPlayer(
			context = context,
			uri = null,
			updateTargetImageView = { bitmap ->
				_state.update { it.copy(mediaPreviewBitmap = bitmap) }
			},
			bitmapFlow = flow,
			onFrame = { bitmap -> 
				Log.v(EyeAIApp.APP_LOG_TAG, "!!! MediaPlayer received frame, submitting to analyzer !!!")
				frameAnalyzer.submitBitmap(bitmap) 
			},
		)
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun stopVideoSource() {
		cameraManager.stop()
		mediaPlayerValue?.shutdown()
		mediaPlayerValue = null
		eyeAIVisionValue?.close()
		eyeAIVisionValue = null
		bitmapFlowValue = null
	}

	internal fun runDepthInference(frame: Bitmap): DepthInferenceResult? = modelLock.read {
		val model = metricDepthModelValue ?: return@read null
		val outputs = model.predictDepth(frame)
		DepthInferenceResult(
			prediction = outputs.legacyMetricDepth,
			rawRelativeDepth = outputs.rawRelativeDepth,
			inputDim = model.inputDim,
			modelName = model.name,
		)
	}

	/**
	 * Runs the frozen REL2ABS Z1 visual head over the raw representation emitted by
	 * the standard MiDaS-v2.1-small model. This intentionally returns null for
	 * another depth model: using an unverified raw representation would violate
	 * the frozen feature contract. V6 gate modes return this visual map and add
	 * their object-level F1 fusion in [MetricDistanceResolver].
	 */
	internal fun runRel2AbsInference(
		frame: Bitmap,
		depthInference: DepthInferenceResult,
	): Rel2AbsRunner.Output? {
		if (depthInference.modelName != EyeAIApp.DEFAULT_DEPTH_MODEL_NAME) {
			Log.w(
				EyeAIApp.APP_LOG_TAG,
				"REL2ABS unavailable for unverified depth model ${depthInference.modelName}",
			)
			return null
		}
		val requestedMode = settings.rel2AbsMode
		return try {
			val output = rel2AbsRunner.run(
				rgbFrame = frame,
				rawRelativeDepth = depthInference.rawRelativeDepth,
				rawWidth = depthInference.inputDim.width,
				rawHeight = depthInference.inputDim.height,
				mode = requestedMode,
			)
			if (settings.rel2AbsMode == requestedMode) output else null
		} catch (error: Throwable) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Frozen REL2ABS inference failed", error)
			null
		}
	}

	internal fun runObjectInference(frame: Bitmap): Array<UniffiDetectedObject>? =
		yoloModel.runInference(frame)

	internal fun runSegmentationInference(frame: Bitmap): SegmentationInferenceResult? {
		val output = segmentationModel.runInference(frame) ?: return null
		return SegmentationInferenceResult(
			prediction = output,
			inputDim = Size(segmentationModel.tensorWidth, segmentationModel.tensorHeight),
			classColors = segmentationModel.classColors,
			modelName = segmentationModel.info.tfliteFilename
		)
	}

	internal suspend fun runOcrInference(frame: Bitmap) = ocrModel.analyzeFrame(frame)

	private fun switchDepthModel(modelName: String) {
		modelLock.write {
			if (metricDepthModelValue?.name == modelName && metricDepthModelValue?.enableNpu == settings.enableNpu) return@write
			metricDepthModelValue = findDepthModelInfo(modelName).createDepthModel(
				context,
				npuQnnDelegateDirectory,
				settings.enableNpu,
			)
		}
	}

	private fun switchNlpModel(modelId: String) {
		val modelInfo = NLPModelInfo.findById(modelId)
		if (nlpModel.info.id == modelInfo.id && nlpModel.isInitialized) return
		nlpModel.create(context, modelInfo)
	}

	private fun findDepthModelInfo(modelName: String): MetricDepthModelInfo =
		EyeAIApp.DEPTH_MODELS.find { it.name == modelName }
			?: EyeAIApp.DEPTH_MODELS.first { it.name == EyeAIApp.DEFAULT_DEPTH_MODEL_NAME }

	private fun hasRecordAudioPermission(): Boolean = ContextCompat.checkSelfPermission(
		context,
		Manifest.permission.RECORD_AUDIO,
	) == PackageManager.PERMISSION_GRANTED

	private fun pauseSpatialAudio() {
		uniffi.NativeLib.setObjectAudioPaused(true)
		uniffi.NativeLib.setDepthAudioPaused(true)
	}

	fun restoreSpatialAudioFromSettings(trigger: String) {
		val current = settings
		uniffi.NativeLib.setObjectAudioPaused(!current.effectiveObjectAudioPlayback)
		uniffi.NativeLib.setDepthAudioPaused(!current.effectiveDepthAudioPlayback)
		Log.i(
			EyeAIApp.APP_LOG_TAG,
			"[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=RESTORED " + "objectAudioEnabled=${current.effectiveObjectAudioPlayback} " + "depthAudioEnabled=${current.effectiveDepthAudioPlayback}",
		)
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun stopAfterStartFailure() {
		stopOperation()
	}

	private inline fun cleanupStep(name: String, action: () -> Unit) {
		try {
			action()
		} catch (error: Throwable) {
			Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI cleanup failed for $name", error)
		}
	}

	private companion object {
		const val INPUT_START_GRACE_NANOS = 15_000_000_000L
		const val INPUT_STALE_TIMEOUT_NANOS = 5_000_000_000L
		const val CAMERA_RESTART_COOLDOWN_NANOS = 10_000_000_000L
	}

	/** Releases all runtime resources; only the Application calls this at shutdown. */
	@RequiresApi(Build.VERSION_CODES.P)
	fun close() {
		synchronized(stateLock) {
			if (runtimeClosed) return
		}
		stopOperation()
		synchronized(stateLock) {
			if (runtimeClosed) return
			runtimeClosed = true
		}
		spatialAudioResumeController.cancel()
		frameAnalyzer.shutdown()
		localSceneDescriptions.close()
		cameraManager.shutdown()
		voskModel.closeService()
		synchronized(stateLock) {
			textToSpeechInstanceValue?.shutdown()
			textToSpeechInstanceValue = null
			rel2AbsRunnerValue?.close()
			rel2AbsRunnerValue = null
			rel2AbsNeuralGateRunnerValue = null
		}
		modelLock.write {
			metricDepthModelValue = null
		}
		nlpModel.close()
		ocrModel.close()
		if (app.localSettingsParserLazyIsInitialized()) app.localSettingsParser.close()
		runtimeJob.cancel()
		speechThreadExecutor.shutdownNow()
		modelExecutor.shutdownNow()
	}
}
