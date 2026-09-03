package com.algorithmic_alliance.eyeaiapp.UI

import android.app.Application
import android.os.Build
import android.os.Looper
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import android.Manifest
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.graphics.createBitmap
import androidx.core.net.toUri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.Settings
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudio
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudioResumeController
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudioResumeOutcome
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.connectivity.EyeAIVision
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.GenericCancellation
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateMachine
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.VoskRestartPolicy
import com.algorithmic_alliance.eyeaiapp.media.MediaPlayer
import com.algorithmic_alliance.eyeaiapp.vibrate
import com.squareup.wire.internal.encodeArray_int32
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(UIState())
    val uiState: StateFlow<UIState> = _uiState.asStateFlow()


    private var currentState: State = State.IDLE
    private var lastFinalResultMillis = System.currentTimeMillis()
    private var llmThreadExecutor = Executors.newSingleThreadExecutor()

    private val spatialAudioResumeController = SpatialAudioResumeController(
        scope = viewModelScope,
        pauseSpatialAudio = ::pauseSpatialAudio,
        restoreSpatialAudio = ::restoreSpatialAudioFromSettings,
        awaitTtsSilence = {
            eyeAIApp().textToSpeechInstance.awaitSilence(
                quietMs = 500L,
                maxWaitMs = 30_000L
            )
        },
        isListening = { eyeAIApp().voskUserStart.get() },
        onOutcome = { trigger, outcome ->
            when (outcome) {
                SpatialAudioResumeOutcome.RESTORED -> Unit
                SpatialAudioResumeOutcome.TTS_SILENCE_TIMEOUT -> Log.w(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=SKIPPED " +
                            "reason=TTS_SILENCE_TIMEOUT"
                )

                SpatialAudioResumeOutcome.LISTENING_STATE_CHANGED -> Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=SKIPPED " +
                            "reason=LISTENING_STATE_CHANGED"
                )
            }
        }
    )

    @RequiresApi(Build.VERSION_CODES.P)
    fun onEvent(event: UIEvent) {
        when (event) {
            is UIEvent.VoskListeningChanged -> {
                Log.d(LOG_TAG, "[MainViewModel] VoskListeningChanged")
                State.IDLE

                if (eyeAIApp().textToSpeechInstance.isSpeaking()) {
                    eyeAIApp().textToSpeechInstance.stop()
                    updateSpeechResponseText("")
                    updateVoskStatusText()
                    setTTSSpeaking(false)
                } else if (!eyeAIApp().voskUserStart.get()) {
                    setTTSSpeaking(false)
                    startVosk()
                } else {
                    stopVosk()
                }
            }

            UIEvent.UpdateVoskStatusText -> {
                Log.d(LOG_TAG, "[MainViewModel] UpdateVoskStatusText")
                updateVoskStatusText()
            }

            UIEvent.OnReloadSettingsPage -> {
                Log.d(LOG_TAG, "[MainViewModel] OnReloadSettingsPage")
                reloadSettingsPage()
            }

            UIEvent.InitVoskService -> {
                Log.d(LOG_TAG, "[MainViewModel] InitVoskService")
                initVoskService()
            }

            UIEvent.CloseVoskService -> {
                Log.d(LOG_TAG, "[MainViewModel] CloseVoskService")
                closeVoskService()
            }

            UIEvent.OnReloadDebugPage -> {
                Log.d(LOG_TAG, "[MainViewModel] ReloadDebugPage")
                reloadDebugPage()
            }

            UIEvent.UpdateSettings -> {
                Log.d(LOG_TAG, "[MainViewModel] UpdateSettings")
                updateSettings()
            }

            is UIEvent.OnUpdatePermissionTutorialCompleted -> {
                Log.d(
                    LOG_TAG,
                    "[MainViewModel] OnUpdatePermissionTutorialCompleted : ${event.value}"
                )
                setPermissionTutorialCompleted(event.value)
            }

            is UIEvent.OnUpdateConnectionTutorialCompleted -> {
                Log.d(
                    LOG_TAG,
                    "[MainViewModel] OnUpdateConnectionTutorialCompleted : ${event.value}"
                )
                setConnectionTutorialCompleted(event.value)
            }

            UIEvent.UpdateSpeechStatusText -> {
                Log.d(LOG_TAG, "[MainViewModel] UpdateLlmStatusText")
            }

            UIEvent.OnOpenSettings -> {
                Log.d(LOG_TAG, "[MainViewModel] OnOpenSettings")
                onOpenSettings()
            }

            UIEvent.OnReturnFromSettings -> {
                Log.d(LOG_TAG, "[MainViewModel] OnReturnFromSettings")
                onReturnFromSettings()
            }

            is UIEvent.OnUpdateSettingsOpened -> {
                Log.d(LOG_TAG, "[MainViewModel] OnUpdateSettingsOpened: ${event.value}")
                _uiState.update { it.copy(settingsOpened = event.value) }
            }

            is UIEvent.OnUpdateAppMissingCameraPermission -> {
                Log.d(LOG_TAG, "[MainViewModel] OnUpdateAppMissingCameraPermission: ${event.value}")
                setAppMissingCameraPermission(event.value)
            }

            is UIEvent.OnUpdateAppMissingVoskPermission -> {
                Log.d(LOG_TAG, "[MainViewModel] OnUpdateAppMissingVoskPermission: ${event.value}")
                setAppMissingVoskPermission(event.value)
            }

            is UIEvent.UIinitCamera -> {
                Log.d(LOG_TAG, "[MainViewModel] UIInitCamera")
                initCamera(event.previewView, event.lifecycleOwner)
            }

            is UIEvent.OnUpdateActionStartedFromSettings -> {
                Log.d(
                    LOG_TAG,
                    "[MainViewModel] OnUpdateConnectionPageStartedFromSettings: ${event.value}"
                )
                _uiState.update { it.copy(actionStartedFromSettings = event.value) }
                Log.d(LOG_TAG, "[MainViewModel] Finished")
            }

            is UIEvent.OnUpdateTTSSpeaking -> {
                setTTSSpeaking(event.value)
            }

            is UIEvent.OnUpdateAppMissingSelectedMediaSource -> {
                setAppMissingSelectedMediaSource(event.value)
            }
        }
    }

    fun setTTSSpeaking(value: Boolean) {
        Log.d(LOG_TAG, "[setLLMSpeaking] : $value")
        _uiState.update { it.copy(ttsSpeaking = value) }
    }

    fun setVoskListening(value: Boolean) {
        _uiState.update { it.copy(voskListening = value) }
    }

    private fun setPermissionTutorialCompleted(value: Boolean) {
        _uiState.update { it.copy(permissionTutorialCompleted = value) }
    }

    private fun setConnectionTutorialCompleted(value: Boolean) {
        _uiState.update { it.copy(connectionTutorialCompleted = value) }
    }

    private fun setAppMissingSelectedMediaSource(value: Boolean) {
        _uiState.update { it.copy(appMissingSelectedMediaSource = value) }
    }


    private fun setAppMissingCameraPermission(value: Boolean) {
        _uiState.update { it.copy(appMissingCameraPermission = value) }
    }

    private fun reloadDebugPage() {
        _uiState.update { it.copy(reloadDebugPageKey = it.reloadDebugPageKey + 1) }
    }

    private fun reloadSettingsPage() {
        _uiState.update { it.copy(reloadSettingsPageKey = it.reloadSettingsPageKey + 1) }
    }

    private fun setAppMissingVoskPermission(value: Boolean) {
        _uiState.update { it.copy(appMissingVoskPermission = value) }
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun onOpenSettings() {
        spatialAudioResumeController.cancel()
        SpatialAudio.stop()
        stopVoskListening()
        eyeAIApp().textToSpeechInstance.stop()

        eyeAIApp().cameraManager.pauseAnalyzer()
        eyeAIApp().mediaFrameAnalyzer?.shutdown()
        eyeAIApp().mediaPlayer?.shutdown()

        uniffi.NativeLib.setDepthAudioPaused(true)
        uniffi.NativeLib.setObjectAudioPaused(true)
    }

    private fun onReturnFromSettings() {
        spatialAudioResumeController.cancel()
        eyeAIApp().aiData.detectedObjects.set(emptyArray())
        startSpatialAudio()
    }

    private fun startSpatialAudio() {
        Log.d("Spatial Audio", "[SpatialAudio] Starting spatial audio")
        Log.d(LOG_TAG, "[SpatialAudio] Starting spatial audio")
        SpatialAudio.setup(eyeAIApp())
        SpatialAudio.start()
        restoreSpatialAudioFromSettings("SPATIAL_AUDIO_START")
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun startVosk() {
        Log.d(LOG_TAG, "[MainViewModel.startVosk] StartVosk called")
        startVoskListening()
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun startVoskListening(trigger: String = "USER_BUTTON") {
        Log.d(LOG_TAG, "[MainViewModel.startVoskListening] StartVoskListening called")
        if (eyeAIApp().voskUserStart.get()) return // Check whether already started
        spatialAudioResumeController.cancel()
        pauseSpatialAudio()

        eyeAIApp().voskUserStart.set(true)
        eyeAIApp().voskModel.startListening()
        Log.i(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][Vosk][START] trigger=$trigger outcome=LISTENING"
        )
        Log.d(EyeAIApp.APP_LOG_TAG, "User started Vosk Model")
        updateVoskStatusText()
        _uiState.update { it.copy(voskListening = true) }
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun stopVosk() {
        Log.d(LOG_TAG, "[MainViewModel.stopVosk] StopVosk called")
        eyeAIApp().textToSpeechInstance.stop()
        android.os.Handler(Looper.getMainLooper()).postDelayed({
            stopVoskListening()
        }, 100)

    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun stopVoskListening(
        trigger: String = "USER_BUTTON",
        spatialAudioResume: SpatialAudioResume = SpatialAudioResume.IMMEDIATE
    ) {
        Log.d(LOG_TAG, "[MainViewModel.stopVoskListening] StopVoskListening called")
        if (!eyeAIApp().voskUserStart.get()) return // Check whether already stopped

        eyeAIApp().voskUserStart.set(false)
        eyeAIApp().voskModel.stopListening()

        when (spatialAudioResume) {
            SpatialAudioResume.IMMEDIATE -> {
                spatialAudioResumeController.cancel()
                restoreSpatialAudioFromSettings(trigger)
            }

            SpatialAudioResume.AFTER_TTS -> spatialAudioResumeController.schedule(trigger)
        }

        Log.d(EyeAIApp.APP_LOG_TAG, "User stopped Vosk Model")
        updateVoskStatusText()
        Log.i(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][Vosk][STOP] trigger=$trigger outcome=STOPPED "
        )
        _uiState.update { it.copy(voskListening = false) }
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun initVoskService() {
        if (!eyeAIApp().voskModel.isListening()) {
            eyeAIApp()
                .voskModel
                .initService(
                    ::onPartialSpeechRecognitionResult,
                    ::onFinalSpeechRecognitionResult,
                    ::onSpeechRecognitionLoaded,
                    { status ->
                        _uiState.update { it.copy(voskListening = status) }
                    }
                )
        }
    }

    private fun closeVoskService() {
        eyeAIApp().voskModel.closeService()
    }

    @RequiresApi(Build.VERSION_CODES.P)
    fun updateVoskStatusText() {
        var voskText = ""
        when {
            !hasPermission(Manifest.permission.RECORD_AUDIO) -> {
                voskText = "Mikrophon-Berechtigung erforderlich"
            }

            !eyeAIApp().settings.enableSpeechRecognition -> {
                voskText = "Spracherkennung deaktiviert"
            }

            eyeAIApp().voskUserStart.get() -> {
                voskText = eyeAIApp().getString(R.string.speech_recognition_ready)
                _uiState.update { it.copy(voskListening = true) }
            }

            else -> {
                voskText = "Vosk bereit - Button klicken zum Starten"
            }
        }

        _uiState.update {
            it.copy(
                speechRecognitionFinalResultText = voskText
            )
        }
    }

    private fun onPartialSpeechRecognitionResult(partial: String) {
        _uiState.update { it.copy(speechRecognitionPartialResultText = partial) }
        if (partial != "")
            Log.d(UI_LOG_TAG, "[MainActivity.onPartialSpeechRecognitionResult] $partial")
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun onSpeechRecognitionLoaded() {
        updateVoskStatusText()
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun onFinalSpeechRecognitionResult(final: String) {
        if (final.isEmpty()) {
            return
        }

        val receiveTs = System.nanoTime()
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][Vosk][RECOGNIZED] originalText='${final.take(200)}' " +
                    "next=STATE_MACHINE currentState=$currentState"
        )


        CoroutineScope(Dispatchers.Main).launch {
            //speechRecognitionFinalResultText?.text = final
            _uiState.update { it.copy(speechRecognitionFinalResultText = final) }
            // minimum of 1 second pause between speech commands
            if (System.currentTimeMillis() - lastFinalResultMillis <= 1000)
                return@launch

            lastFinalResultMillis = System.currentTimeMillis()

            // Pause listening while the local state machine evaluates and speaks.
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][Vosk][PAUSE_FOR_PROCESSING] autoRestartAfterTts=true"
            )
            eyeAIApp().voskModel.stopListening()
            _uiState.update { it.copy(voskListening = false) }

            // vibrate for 100ms
            vibrate(eyeAIApp(), 100)

            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][DISPATCH] state=$currentState; latencySinceVosk=${
                    elapsedMs(receiveTs)
                }ms"
            )

            withContext(eyeAIApp().speechThreadExecutor.asCoroutineDispatcher()) {
                val workerStart = System.nanoTime()
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][WORKER] phase=START state=$currentState"
                )
                onSpeechResult(final)
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][WORKER] phase=FINISH duration=${
                        elapsedMs(workerStart)
                    }ms"
                )
            }
        }
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private suspend fun onSpeechResult(final: String) {
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][INPUT] state=$currentState originalText='$final'"
        )
        setTTSSpeaking(true)
        val stateMachine = StateMachine(
            eyeAIApp(),
            eyeAIApp().textToSpeechInstance,
            eyeAIApp().lastDialogContext,
            setSpeechResponseText = { string -> _uiState.update { it.copy(speechResponseText = string) } },
            eyeAIApp().cameraManager.cameraFrameAnalyzer ?: eyeAIApp().mediaFrameAnalyzer
        )

        val cancellationResponse = GenericCancellation.responseFor(final)
        val update = if (cancellationResponse != null) {
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][CANCEL] input matched generic cancellation before state dispatch"
            )
            stateMachine.handleCancellation()
        } else {
            when (currentState) {
                State.IDLE -> stateMachine.handleIdle(final)
                State.SETTINGS_MENU -> stateMachine.handleSettingsMenu(final)
                State.SETTINGS_CHOICE -> stateMachine.handleSettingsChoice(final)
                State.SETTINGS_ACTION -> stateMachine.handleSettingsAction(final)
                State.SETTINGS_EXTERNAL_CONFIRMATION ->
                    stateMachine.handleSettingsExternalConfirmation(final)
            }
        }
        if (update.voskRestartPolicy == VoskRestartPolicy.REQUIRE_MANUAL_RESTART) {
            Log.i(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][Vosk][POLICY] source=SETTINGS_APPLIED " +
                        "policy=REQUIRE_MANUAL_RESTART"
            )
            stopVoskListening(
                trigger = "SETTINGS_APPLIED",
                spatialAudioResume = SpatialAudioResume.AFTER_TTS
            )
        }

        // Logging der state transition
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][TRANSITION] $currentState -> ${update.newState}; " +
                    "voskRestartPolicy=${update.voskRestartPolicy}"
        )
        currentState = update.newState
        eyeAIApp().lastDialogContext = update.newJson
    }

    fun updateSpeechResponseText(text: String) {
        _uiState.update { it.copy(speechResponseText = text) }
    }

    private fun hasPermission(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(
            eyeAIApp(),
            permission
        ) == PackageManager.PERMISSION_GRANTED
    }

    @RequiresApi(Build.VERSION_CODES.P)
    fun onResume() {
        Log.d(LOG_TAG, "[MainViewModel] OnResume")
        if (_uiState.value.actionStartedFromSettings || _uiState.value.settingsOpened) {
            Log.d(LOG_TAG, "[MainViewModel] Exited OnResume")
            return
        }

        reloadDebugPage()
        if (!hasPermission(Manifest.permission.CAMERA) && _uiState.value.permissionTutorialCompleted) {
            setAppMissingCameraPermission(true)
        }
        if (eyeAIApp().settings.enableSpeechRecognition && !hasPermission(Manifest.permission.RECORD_AUDIO) && _uiState.value.permissionTutorialCompleted) {
            setAppMissingVoskPermission(true)
        }
        updateVoskStatusText()

        CoroutineScope(Dispatchers.IO).launch {
            Log.d("Spatial Audio", "[SpatialAudio] Starting spatial audio")
            SpatialAudio.setup(eyeAIApp())
            SpatialAudio.start()
        }

        if (
            eyeAIApp().voskUserStart.get() ||
            spatialAudioResumeController.isPending() ||
            eyeAIApp().textToSpeechInstance.isSpeaking()
        ) {
            pauseSpatialAudio()
        } else {
            restoreSpatialAudioFromSettings("ON_RESUME")
        }
    }

    fun onPause() {
        spatialAudioResumeController.cancel()
        pauseSpatialAudio()
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun initCamera(cameraPreviewView: PreviewView?, lifecycleOwner: LifecycleOwner) {
        _uiState.update { it.copy(detectedObjects = emptyArray()) }
        _uiState.update { it.copy(ocrResults = emptyArray()) }
        _uiState.update { it.copy(depthPreviewBitmap = createBitmap(256, 256)) }
        if (eyeAIApp().settings.inputSource == eyeAIApp().getString(R.string.input_is_camera)) {
            if (hasPermission(Manifest.permission.CAMERA)) {
                eyeAIApp().cameraManager.cameraFrameAnalyzer?.shutdown()
                eyeAIApp().cameraManager.cameraFrameAnalyzer =
                    CameraFrameAnalyzer(
                        eyeAIApp(),
                        { bitmap -> _uiState.update { it.copy(depthPreviewBitmap = bitmap) } },
                        //performanceText!!,
                        { string -> _uiState.update { it.copy(performanceText = string) } },
                        //overlayObjectDetection!!,
                        { results -> _uiState.update { it.copy(detectedObjects = results) } },
                        { size -> _uiState.update { it.copy(cameraResolution = size) } },
                        { bitmap -> _uiState.update { it.copy(debugInputPreviewBitmap = bitmap) } },
                        { _uiState.value.mediaPreviewBitmap }
                    )
                eyeAIApp().cameraManager.cameraFrameAnalyzer?.start()

                eyeAIApp().cameraManager
                    .init(
                        eyeAIApp(),
                        lifecycleOwner,
                        EyeAIApp.PREFERRED_CAMERA_RESOLUTION,
                        cameraPreviewView,
                    )
            } else {
                _uiState.update { it.copy(appMissingCameraPermission = true) }
            }
        } else if (eyeAIApp().settings.inputSource == eyeAIApp().getString(R.string.input_is_media)) {
            if (eyeAIApp().settings.mediaSource!!.isNotEmpty()) {
                Log.d(
                    LOG_TAG,
                    "[MainViewModel.initCamera] Input source is media and MediaSource ${eyeAIApp().settings.mediaSource!!.toUri()} is not empty"
                )
                ProcessCameraProvider.getInstance(eyeAIApp()).get().unbindAll()
                _uiState.update { it.copy(detectedObjects = emptyArray()) }
                _uiState.update { it.copy(ocrResults = emptyArray()) }
                _uiState.update { it.copy(depthPreviewBitmap = createBitmap(256, 256)) }


                eyeAIApp().mediaPlayer?.shutdown()
                eyeAIApp().mediaPlayer =
                    MediaPlayer(
                        eyeAIApp(),
                        eyeAIApp().settings.mediaSource!!.toUri(),
                        { bitmap -> _uiState.update { it.copy(mediaPreviewBitmap = bitmap) } }
                    )


                eyeAIApp().mediaFrameAnalyzer?.shutdown()
                eyeAIApp().mediaFrameAnalyzer =
                    CameraFrameAnalyzer(
                        eyeAIApp(),
                        { bitmap -> _uiState.update { it.copy(depthPreviewBitmap = bitmap) } },
                        { string -> _uiState.update { it.copy(performanceText = string) } },
                        { results -> _uiState.update { it.copy(detectedObjects = results) } },
                        { size -> _uiState.update { it.copy(cameraResolution = size) } },
                        { bitmap -> _uiState.update { it.copy(debugInputPreviewBitmap = bitmap) } },
                        { _uiState.value.mediaPreviewBitmap }
                    )

                eyeAIApp().mediaFrameAnalyzer?.start()
            } else {
                Log.d(
                    LOG_TAG,
                    "[MainViewModel.initCamera] Input Source is media but selected file is empty"
                )
                setAppMissingSelectedMediaSource(true)
            }

        } else if (eyeAIApp().settings.inputSource == eyeAIApp().getString(R.string.input_is_eyeaivision)) {
            if (!eyeAIApp().settings.eyeAIVisionIP!!.isEmpty()) {
                eyeAIApp().bitmapFlow = MutableSharedFlow(replay = 1)

                /* TODO reimplement
                val connectingTCPDialog = AlertDialog.Builder(eyeAIApp())
                connectingTCPDialog.setMessage("Connecting to Button Server...")
                connectingTCPDialog.setView(ProgressBar(eyeAIApp()))

                var shownConnectDialog: AlertDialog? = null
                */
                eyeAIApp().eyeAIVision = EyeAIVision(
                    ip = eyeAIApp().settings.eyeAIVisionIP.toString(),
                    eyeAIApp().settings.jpegCompression,
                    lifecycleScope = viewModelScope,
                    bitmapFlow = eyeAIApp().bitmapFlow,
                    onSingleClick = {
                        Log.i("CLICK", "SINGLE")

                        State.IDLE

                        if (!eyeAIApp().voskUserStart.get()) {
                            startVoskListening(trigger = "EYEAIVISION_BUTTON")
                        }


                    },
                    onDoubleClick = {
                        Log.i("CLICK", "DOUBLE")

                        State.IDLE

                        if (eyeAIApp().voskUserStart.get()) {
                            eyeAIApp().textToSpeechInstance.stop()
                            stopVoskListening()
                        }
                    },
                    onSocketFailed = { e ->
                        // TODO reimplement
                        /*
                        if (!tcpErrorShowing) {
                            tcpErrorShowing = true
                            val errorMessage = AlertDialog.Builder(this)
                            errorMessage.setMessage("TCP connection to EyeAIVision (IP: ${eyeAIApp().settings.eyeAIVisionIP.toString()}) has failed: ${e.message.toString()}")
                            errorMessage.setPositiveButton("Open settings") { dialog, which ->
                                tcpErrorShowing = false
                                startActivity(Intent(this, SettingsActivity::class.java))
                                dialog.dismiss()
                                overridePendingTransition(
                                    android.R.anim.fade_in,
                                    android.R.anim.fade_out
                                )
                            }

                            errorMessage.setNegativeButton("Ignore") { dialog, which ->
                                tcpErrorShowing = false
                                dialog.dismiss()
                            }
                            errorMessage.show()
                        }
                         */

                    },

                    onMjpegError = { e ->
                        /* TODO reimplement
                        runOnUiThread {
                            if (!mjpegErrorShowing && !mjpegErrorIgnored) {
                                mjpegErrorShowing = true
                                val errorMessage = AlertDialog.Builder(this)
                                errorMessage.setMessage("Error while getting camera frame from EyeAIVision (IP: ${eyeAIApp().settings.eyeAIVisionIP.toString()}): ${e.message.toString()}")
                                errorMessage.setPositiveButton("Open settings") { dialog, which ->
                                    mjpegErrorShowing = false
                                    dialog.dismiss()
                                    startActivity(Intent(this, SettingsActivity::class.java))
                                    overridePendingTransition(
                                        android.R.anim.fade_in,
                                        android.R.anim.fade_out
                                    )
                                }

                                errorMessage.setNegativeButton("Ignore") { dialog, which ->
                                    dialog.dismiss()
                                    mjpegErrorIgnored = true
                                    mjpegErrorShowing = false
                                }
                                errorMessage.show()
                            }
                        }
                         */
                    },

                    onConnectingSocket = {
                        /* TODO reimplement
                        runOnUiThread {
                            shownConnectDialog = connectingTCPDialog.show()
                        }
                         */
                    },

                    onSocketConnectionEstablished = {
                        /* TODO reimplement
                        runOnUiThread {
                            shownConnectDialog?.dismiss()
                        }
                         */
                    },

                    onConnectingHTTP = {

                    },

                    onHTTPConnectionEstablished = {

                    }
                )

                eyeAIApp().mediaPlayer?.shutdown()

                eyeAIApp().mediaPlayer = MediaPlayer(
                    context = eyeAIApp(),
                    uri = null,
                    { bitmap -> _uiState.update { it.copy(mediaPreviewBitmap = bitmap) } },
                    bitmapFlow = eyeAIApp().bitmapFlow
                )





                eyeAIApp().mediaFrameAnalyzer?.shutdown()
                eyeAIApp().mediaFrameAnalyzer = CameraFrameAnalyzer(
                    eyeAIApp(),
                    { bitmap -> _uiState.update { it.copy(depthPreviewBitmap = bitmap) } },
                    //performanceText!!,
                    { string -> _uiState.update { it.copy(performanceText = string) } },
                    //overlayObjectDetection!!,
                    { results -> _uiState.update { it.copy(detectedObjects = results) } },
                    { size -> _uiState.update { it.copy(cameraResolution = size) } },
                    { bitmap -> _uiState.update { it.copy(debugInputPreviewBitmap = bitmap) } },
                    { _uiState.value.mediaPreviewBitmap }
                )
                eyeAIApp().mediaFrameAnalyzer?.start()

            } else {
                /* TODO reimplement
                val builder = AlertDialog.Builder(this)
                builder.setMessage("No IP address has been entered. Please enter one in the settings menu")
                    .setPositiveButton("Open settings") { dialog, id ->
                        startActivity(Intent(this, SettingsActivity::class.java))
                        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                    }
                builder.create().show()

                 */
            }
        }
    }

    fun updateSettings() {
        eyeAIApp().updateSettings()
    }

    fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

    private fun pauseSpatialAudio() {
        uniffi.NativeLib.setObjectAudioPaused(true)
        uniffi.NativeLib.setDepthAudioPaused(true)
    }

    private fun restoreSpatialAudioFromSettings(trigger: String) {
        val settings = Settings.load(eyeAIApp())
        uniffi.NativeLib.setObjectAudioPaused(!settings.objectAudioPlayback)
        uniffi.NativeLib.setDepthAudioPaused(!settings.depthAudioPlayback)
        Log.i(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=RESTORED " +
                    "objectAudioEnabled=${settings.objectAudioPlayback} " +
                    "depthAudioEnabled=${settings.depthAudioPlayback}"
        )
    }

    private enum class SpatialAudioResume {
        IMMEDIATE,
        AFTER_TTS
    }

    fun eyeAIApp(): EyeAIApp {
        return getApplication()
    }
}
