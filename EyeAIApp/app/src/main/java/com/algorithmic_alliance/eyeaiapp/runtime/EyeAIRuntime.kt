package com.algorithmic_alliance.eyeaiapp.runtime

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.camera.view.PreviewView
import androidx.annotation.RequiresApi
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.Settings
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudio
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudioResumeController
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudioResumeOutcome
import com.algorithmic_alliance.eyeaiapp.audio.AudioFrame
import com.algorithmic_alliance.eyeaiapp.audio.AudioFrameSink
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.camera.FrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.camera.FrameAnalysisUpdate
import com.algorithmic_alliance.eyeaiapp.connectivity.EyeAIVision
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModel
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModel
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModelInfo
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.EyeAIState
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.GenericCancellation
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateMachine
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.VoskRestartPolicy
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModel
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModelInfo
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModel
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModelInfo
import com.algorithmic_alliance.eyeaiapp.ocr.GoogleOCR
import com.algorithmic_alliance.eyeaiapp.settingsparser.LocalSettingsParser
import com.algorithmic_alliance.eyeaiapp.speech_recognition.VoskModel
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import com.algorithmic_alliance.eyeaiapp.vibrate
import com.algorithmic_alliance.eyeaiapp.media.MediaPlayer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.lang.ref.WeakReference
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import uniffi.NativeLib.UniffiDetectedObject

/** Output of a depth inference while the model read lock is held. */
data class DepthInferenceResult(
    val prediction: NativeLib.NativeFloatBuffer,
    val inputDim: android.util.Size,
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
    private var textToSpeechInstanceValue: TextToSpeechInstance? = null
    private var speechCallbacksInstalled = false
    private var lastFinalResultMillis = 0L
    private var voskStarting = AtomicBoolean(false)
    private var currentState = EyeAIState.IDLE
    private var lastDialogContextValue: String? = null
    private var runtimeClosed = false
    private var serviceOwner = WeakReference<LifecycleOwner>(null)
    private var audioFrameSink: AudioFrameSink? = null
    private var mediaPlayerValue: MediaPlayer? = null
    private var eyeAIVisionValue: EyeAIVision? = null
    private var bitmapFlowValue: MutableSharedFlow<Bitmap>? = null

    private val spatialAudioResumeController = SpatialAudioResumeController(
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
                    "[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=SKIPPED " +
                        "reason=TTS_SILENCE_TIMEOUT",
                )
                SpatialAudioResumeOutcome.LISTENING_STATE_CHANGED -> Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=SKIPPED " +
                        "reason=LISTENING_STATE_CHANGED",
                )
            }
        },
    )

    val speechThreadExecutorForStateMachine = speechThreadExecutor
    val voskUserStart = AtomicBoolean(false)
    val yoloModel = YoloModel(YoloModelInfo("model.tflite", "coco.names", 640))
    val nlpModel = NLPModel(NLPModelInfo.findById(NLPModelInfo.DEFAULT_MODEL_ID))
    val ocrModel = GoogleOCR()
    val voskModel = VoskModel(context, "model-de")
    val frameAnalyzer = FrameAnalyzer(this, ::onFrameAnalysisUpdate)
    val cameraManager = CameraManager(::onCameraStateChanged)
    val npuQnnDelegateDirectory: String = app.applicationInfo.nativeLibraryDir

    val settings: Settings
        get() = app.settings

    val metricDepthModel: MetricDepthModel?
        get() = modelLock.read { metricDepthModelValue }

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

    internal fun setLastDialogContextFromCompatibility(value: String?) {
        lastDialogContext = value
    }

    val confirmationModel: ConfirmationModel
        get() = app.confirmationModel

    val localSettingsParser: LocalSettingsParser
        get() = app.localSettingsParser

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
        if (oldSettings.depthAudioPlayback != newSettings.depthAudioPlayback) {
            uniffi.NativeLib.setDepthAudioPaused(!newSettings.depthAudioPlayback)
        }
        if (oldSettings.objectAudioPlayback != newSettings.objectAudioPlayback) {
            uniffi.NativeLib.setObjectAudioPaused(!newSettings.objectAudioPlayback)
        }
        if (
            oldSettings.depthAudioFrequency != newSettings.depthAudioFrequency ||
            oldSettings.depthAudioClickIncidence != newSettings.depthAudioClickIncidence
        ) {
            uniffi.NativeLib.setAudioSettings(
                newSettings.depthAudioFrequency.toFloat(),
                newSettings.depthAudioClickIncidence,
            )
        }

        modelScope.launch {
            try {
                if (oldSettings.nlpModel != newSettings.nlpModel) {
                    switchNlpModel(newSettings.nlpModel)
                }
                if (
                    oldSettings.depthModel != newSettings.depthModel ||
                    oldSettings.enableNpu != newSettings.enableNpu
                ) {
                    switchDepthModel(newSettings.depthModel)
                }
                if (
                    newSettings.enableObjectDetection &&
                    (
                        !oldSettings.enableObjectDetection ||
                            oldSettings.enableNpu != newSettings.enableNpu
                        )
                ) {
                    yoloModel.create(context, npuQnnDelegateDirectory, newSettings.enableNpu)
                }
                if (newSettings.enableOCR && !oldSettings.enableOCR) ocrModel.create()
                if (
                    isActive &&
                    oldSettings.objectAudioPlaybackLanguage !=
                        newSettings.objectAudioPlaybackLanguage
                ) {
                    SpatialAudio.setup(context)
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
            } catch (error: Throwable) {
                Log.e(EyeAIApp.APP_LOG_TAG, "Settings-triggered runtime update failed", error)
            }
        }
    }

    /** Starts the active local source under the foreground service lifecycle. */
    fun start(owner: LifecycleOwner) {
        synchronized(stateLock) {
            check(!runtimeClosed) { "EyeAI runtime is closed" }
        }
        if (!lifecycleGate.start()) return

        try {
            initializeModels()
            textToSpeechInstance
            frameAnalyzer.start()
            SpatialAudio.setup(context)
            SpatialAudio.start()
            serviceOwner = WeakReference(owner)
            startVideoSource(owner)
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
    fun stopOperation() {
        if (!lifecycleGate.stop()) return
        cleanupStep("spatial-audio resume controller") {
            spatialAudioResumeController.cancel()
        }
        cleanupStep("frame analyzer") { frameAnalyzer.stop() }
        cleanupStep("video source") { stopVideoSource() }
        voskUserStart.set(false)
        cleanupStep("Vosk listener") { voskModel.stopListening() }
        cleanupStep("TTS") { synchronized(stateLock) { textToSpeechInstanceValue }?.stop() }
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

    /** Programmatic TTS API usable from the service or future hardware adapter. */
    fun speak(text: String, queueMode: Int = TextToSpeechInstance.QUEUE_FLUSH) {
        if (text.isBlank()) return
        _state.update { it.copy(ttsSpeaking = true, speechResponseText = text) }
        textToSpeechInstance.speak(text, queueMode)
    }

    /**
     * Future external audio adapters can attach at this neutral boundary.
     * The current local Vosk/SpeechService path remains the active source.
     */
    fun attachAudioFrameSink(sink: AudioFrameSink?) {
        synchronized(stateLock) { audioFrameSink = sink }
    }

    fun submitAudioFrame(frame: AudioFrame): Boolean {
        val sink = synchronized(stateLock) { audioFrameSink }
        return sink?.submit(frame) == true
    }

    fun toggleListening() {
        if (textToSpeechInstance.isSpeaking()) {
            textToSpeechInstance.stop()
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

    private fun stopListening(
        trigger: String = "USER_BUTTON",
        restoreAfterTts: Boolean = false,
    ) {
        if (!voskUserStart.getAndSet(false)) return
        voskModel.stopListening()
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
            Log.d(EyeAIApp.APP_LOG_TAG, "[Vosk] partial=$partial")
        }
    }

    private fun onSpeechRecognitionLoaded() {
        updateVoskStatusText()
    }

    private fun onFinalSpeechRecognitionResult(final: String) {
        if (final.isEmpty()) return
        val receiveTs = System.nanoTime()
        _state.update {
            it.copy(
                speechRecognitionFinalResultText = final,
                voskListening = false,
            )
        }
        val now = System.currentTimeMillis()
        if (now - lastFinalResultMillis <= 1_000L) return
        lastFinalResultMillis = now

        voskModel.stopListening()
        vibrate(context, 100)
        speechScope.launch {
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][DISPATCH] state=$currentState; " +
                    "latencySinceVosk=${(System.nanoTime() - receiveTs) / 1_000_000}ms",
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
                    EyeAIState.SETTINGS_EXTERNAL_CONFIRMATION ->
                        stateMachine.handleSettingsExternalConfirmation(final)
                }
            }
            if (update.voskRestartPolicy == VoskRestartPolicy.REQUIRE_MANUAL_RESTART) {
                stopListening("SETTINGS_APPLIED", restoreAfterTts = true)
            }
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][TRANSITION] $currentState -> ${update.newState}; " +
                    "voskRestartPolicy=${update.voskRestartPolicy}",
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
                    voskModel.startListening()
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
            !hasRecordAudioPermission() -> "Mikrophon-Berechtigung erforderlich"
            !settings.enableSpeechRecognition -> "Spracherkennung deaktiviert"
            voskUserStart.get() -> context.getString(R.string.speech_recognition_ready)
            else -> "Vosk bereit - Button klicken zum Starten"
        }
        _state.update { it.copy(speechRecognitionFinalResultText = text) }
    }

    private fun publishVoskStatus(status: Boolean) {
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
        when (source) {
            context.getString(R.string.input_is_camera) -> {
                cameraManager.start(
                    context = context,
                    owner = owner,
                    preferredImageSize = EyeAIApp.PREFERRED_CAMERA_RESOLUTION,
                    cameraPreviewView = null,
                    frameAnalyzer = frameAnalyzer,
                )
            }
            context.getString(R.string.input_is_media) -> {
                val mediaSource = settings.mediaSource
                if (mediaSource.isNullOrEmpty()) {
                    _state.update { it.copy(lastError = "Keine Media-Quelle ausgewählt") }
                } else if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
                    _state.update {
                        it.copy(lastError = "Media-Eingabe benötigt Android 9 oder neuer")
                    }
                } else {
                    startMediaSource(Uri.parse(mediaSource))
                }
            }
            context.getString(R.string.input_is_eyeaivision) -> {
                val ip = settings.eyeAIVisionIP
                if (ip.isNullOrEmpty()) {
                    _state.update { it.copy(lastError = "Keine EyeAI-Vision-Adresse ausgewählt") }
                } else if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
                    _state.update {
                        it.copy(lastError = "EyeAI-Vision-Eingabe benötigt Android 9 oder neuer")
                    }
                } else {
                    startEyeAIVisionSource(ip)
                }
            }
            else -> _state.update { it.copy(lastError = "Unbekannte Eingabequelle: $source") }
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
    private fun startEyeAIVisionSource(ip: String) {
        val flow = MutableSharedFlow<Bitmap>(replay = 1, extraBufferCapacity = 1)
        bitmapFlowValue = flow
        eyeAIVisionValue = EyeAIVision(
            ip = ip,
            compression = settings.jpegCompression,
            lifecycleScope = runtimeScope,
            bitmapFlow = flow,
            onSingleClick = { startListening("EYEAIVISION_BUTTON") },
            onDoubleClick = { stopListening("EYEAIVISION_BUTTON") },
            onConnectingSocket = {},
            onSocketConnectionEstablished = {},
            onSocketFailed = { error ->
                _state.update { it.copy(lastError = error.message) }
            },
            onMjpegError = { error ->
                _state.update { it.copy(lastError = error.message) }
            },
            onConnectingHTTP = {},
            onHTTPConnectionEstablished = {},
        )
        mediaPlayerValue = MediaPlayer(
            context = context,
            uri = null,
            updateTargetImageView = { bitmap ->
                _state.update { it.copy(mediaPreviewBitmap = bitmap) }
            },
            bitmapFlow = flow,
            onFrame = { bitmap -> frameAnalyzer.submitBitmap(bitmap) },
        )
    }

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
        DepthInferenceResult(
            prediction = model.predictDepth(frame),
            inputDim = model.inputDim,
            modelName = model.name,
        )
    }

    internal fun runObjectInference(frame: Bitmap): Array<UniffiDetectedObject>? =
        yoloModel.runInference(frame)

    internal suspend fun runOcrInference(frame: Bitmap) = ocrModel.analyzeFrame(frame)

    private fun switchDepthModel(modelName: String) {
        modelLock.write {
            if (
                metricDepthModelValue?.name == modelName &&
                metricDepthModelValue?.enableNpu == settings.enableNpu
            ) return@write
            metricDepthModelValue?.close()
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

    private fun restoreSpatialAudioFromSettings(trigger: String) {
        val current = settings
        uniffi.NativeLib.setObjectAudioPaused(!current.objectAudioPlayback)
        uniffi.NativeLib.setDepthAudioPaused(!current.depthAudioPlayback)
        Log.i(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=RESTORED " +
                "objectAudioEnabled=${current.objectAudioPlayback} " +
                "depthAudioEnabled=${current.depthAudioPlayback}",
        )
    }

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

    /** Releases all runtime resources; only the Application calls this at shutdown. */
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
        cameraManager.shutdown()
        voskModel.closeService()
        synchronized(stateLock) {
            textToSpeechInstanceValue?.shutdown()
            textToSpeechInstanceValue = null
        }
        modelLock.write {
            metricDepthModelValue?.close()
            metricDepthModelValue = null
        }
        nlpModel.close()
        if (app.localSettingsParserLazyIsInitialized()) app.localSettingsParser.close()
        runtimeJob.cancel()
        speechThreadExecutor.shutdownNow()
        modelExecutor.shutdownNow()
    }
}
