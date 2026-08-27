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
import com.algorithmic_alliance.eyeaiapp.MainActivity
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.Settings
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudio
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.connectivity.EyeAIVision
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG
import com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio.SpeechManager
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateMachine
import com.algorithmic_alliance.eyeaiapp.media.MediaPlayer
import com.algorithmic_alliance.eyeaiapp.vibrate
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

    private val voskUserStart = AtomicBoolean(false)

    private var currentState: State = State.IDLE
    private var lastFinalResultMillis = System.currentTimeMillis()
    private var llmThreadExecutor = Executors.newSingleThreadExecutor()

    @RequiresApi(Build.VERSION_CODES.P)
    fun onEvent(event: UIEvent) {
        Log.d(LOG_TAG, "[MainViewModel] onEvent called")
        when (event) {
            is UIEvent.VoskListeningChanged -> {
                Log.d(LOG_TAG, "[MainViewModel] VoskListeningChanged")
                State.IDLE
                if (!voskUserStart.get()) {
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

            UIEvent.UpdateSettings -> {
                Log.d(LOG_TAG, "[MainViewModel] UpdateSettings")
                updateSettings()
            }

            UIEvent.UpdateLlmStatusText -> {
                Log.d(LOG_TAG, "[MainViewModel] UpdateLlmStatusText")
                val isLLMConfigured = eyeAIApp().settings.googleAiStudioApiKey?.isEmpty() == false
                updateLlmResponseText(
                    if (isLLMConfigured)
                        ""
                    else
                        eyeAIApp().getString(R.string.setup_llm_notice)
                )
            }

            UIEvent.OnOpenSettings -> {
                Log.d(LOG_TAG, "[MainViewModel] OnOpenSettings")
                onOpenSettings()
            }

            UIEvent.OnReturnFromSettings -> {
                Log.d(LOG_TAG, "[MainViewModel] OnReturnFromSettings")
                onReturnFromSettings()
            }

            is UIEvent.OnUpdateAppMissingVoskPermission -> {
                setAppMissingVoskPermission(event.value)
            }

            is UIEvent.UIinitCamera -> {
                Log.d(LOG_TAG, "[MainViewModel] UIInitCamera")
                initCamera(event.previewView, event.lifecycleOwner)
            }
        }
    }

    private fun reloadSettingsPage() {
        _uiState.update { it.copy(reloadSettingsPageKey = it.reloadSettingsPageKey + 1) }
    }

    private fun setAppMissingVoskPermission(value: Boolean) {
        _uiState.update { it.copy(appMissingVoskPermission = value) }
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun onOpenSettings() {
        SpatialAudio.stop()
        eyeAIApp().voskModel.stopListening()

        eyeAIApp().cameraManager.pauseAnalyzer()
        eyeAIApp().mediaFrameAnalyzer?.shutdown()
        eyeAIApp().mediaPlayer?.shutdown()

        uniffi.NativeLib.setDepthAudioPaused(true)
        uniffi.NativeLib.setObjectAudioPaused(true)
    }

    private fun onReturnFromSettings() {
        eyeAIApp().aiData.detectedObjects.set(emptyArray())
        startSpatialAudio()
    }

    private fun startSpatialAudio() {
        CoroutineScope(Dispatchers.IO).launch {
            Log.d("Spatial Audio", "[SpatialAudio] Starting spatial audio")
            SpatialAudio.setup(eyeAIApp())
            SpatialAudio.start()
        }
        val settings = Settings.load(eyeAIApp())
        uniffi.NativeLib.setObjectAudioPaused(!settings.objectAudioPlayback)
        uniffi.NativeLib.setDepthAudioPaused(!settings.depthAudioPlayback)
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun startVosk() {
        Log.d(LOG_TAG, "[MainViewModel.startVosk] StartVosk called")
        startVoskListening()
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun startVoskListening() {
        Log.d(LOG_TAG, "[MainViewModel.startVoskListening] StartVoskListening called")
        if (voskUserStart.get()) return // Check whether already started

        uniffi.NativeLib.setObjectAudioPaused(true)
        uniffi.NativeLib.setDepthAudioPaused(true)
        voskUserStart.set(true)
        eyeAIApp().voskModel.startListening()
        Log.d(EyeAIApp.APP_LOG_TAG, "User started Vosk Model")
        updateVoskStatusText()
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun stopVosk() {
        Log.d(LOG_TAG, "[MainViewModel.stopVosk] StopVosk called")
        SpeechManager.forceStop()
        android.os.Handler(Looper.getMainLooper()).postDelayed({
            stopVoskListening()
        }, 100)

    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun stopVoskListening() {
        Log.d(LOG_TAG, "[MainViewModel.stopVoskListening] StopVoskListening called")
        if (!voskUserStart.get()) return // Check whether already stopped

        val settings = Settings.load(eyeAIApp())
        uniffi.NativeLib.setObjectAudioPaused(!settings.objectAudioPlayback)
        uniffi.NativeLib.setDepthAudioPaused(!settings.depthAudioPlayback)
        voskUserStart.set(false)
        eyeAIApp().voskModel.stopListening()
        Log.d(EyeAIApp.APP_LOG_TAG, "User stopped Vosk Model")
        updateVoskStatusText()
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun initVoskService() {
        if (!eyeAIApp().voskModel.isListening()) {
            eyeAIApp()
                .voskModel
                .initService(
                    ::onPartialSpeechRecognitionResult,
                    ::onFinalSpeechRecognitionResult,
                    ::onSpeechRecognitionLoaded
                )
        }
    }

    private fun closeVoskService() {
        eyeAIApp().voskModel.closeService()
    }

    @RequiresApi(Build.VERSION_CODES.P)
    fun updateVoskStatusText() {
        _uiState.update {
            it.copy(
                speechRecognitionFinalResultText = when {
                    !hasPermission(Manifest.permission.RECORD_AUDIO) -> "Mikrophon-Berechtigung erforderlich"
                    !eyeAIApp().settings.enableSpeechRecognition -> "Spracherkennung deaktiviert"
                    voskUserStart.get() -> eyeAIApp().getString(R.string.speech_recognition_ready)
                    else -> "Vosk bereit - Button klicken zum Starten"
                }
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

    private fun onFinalSpeechRecognitionResult(final: String) {
        if (final.isEmpty()) {
            return
        }

        val receiveTs = System.nanoTime()
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "SR final RECEIVED at ${System.currentTimeMillis()} (ms), text='${final.take(200)}'"
        )


        CoroutineScope(Dispatchers.Main).launch {
            _uiState.update { it.copy(speechRecognitionFinalResultText = final) }

            // minimum of 1 second pause between speech commands
            if (System.currentTimeMillis() - lastFinalResultMillis <= 1000)
                return@launch

            lastFinalResultMillis = System.currentTimeMillis()

            if (eyeAIApp().llm == null) {
                if (eyeAIApp().settings.enableSpeechRecognition) {
                    _uiState.update { it.copy(llmResponseText = eyeAIApp().getString(R.string.setup_llm_notice)) }
                    //llmResponseText?.text = getString(R.string.setup_llm_notice)
                }
            } else {
                _uiState.update { it.copy(llmResponseText = eyeAIApp().getString(R.string.llm_responding_notice)) }
                //llmResponseText?.text = getString(R.string.llm_responding_notice)

                //start after onTTSFinished speaking
                //Logging when Vosk is stopped.
                Log.d(EyeAIApp.APP_LOG_TAG, "Stopping Vosk to process command.")
                eyeAIApp().voskModel.stopListening()

                // vibrate for 100ms
                vibrate(eyeAIApp(), 100)

                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "Dispatching to LLM worker at ${System.currentTimeMillis()} (ms); latency since SR receive = ${
                        elapsedMs(receiveTs)
                    } ms"
                )

                withContext(llmThreadExecutor.asCoroutineDispatcher()) {
                    val workerStart = System.nanoTime()
                    Log.d(
                        EyeAIApp.APP_LOG_TAG,
                        "LLM worker START processing at ${System.currentTimeMillis()} (ms)"
                    )
                    onSpeechResult(final)
                    Log.d(
                        EyeAIApp.APP_LOG_TAG,
                        "LLM worker FINISHED processing at ${System.currentTimeMillis()} (ms); duration=${
                            elapsedMs(workerStart)
                        } ms"
                    )
                }

            }
        }
    }

    private suspend fun onSpeechResult(final: String) {
        Log.d(EyeAIApp.APP_LOG_TAG, "onSpeechResult: Creating new StateMachine for input: '$final'")

        val stateMachine = StateMachine(
            eyeAIApp(),
            eyeAIApp().textToSpeechInstance,
            eyeAIApp().lastLlmJsonResponse,
            { text -> _uiState.update { it.copy(llmResponseText = text) } },
            { text -> _uiState.update { it.copy(llmResponseText = it.llmResponseText + text) } },
            eyeAIApp().cameraManager.cameraFrameAnalyzer ?: eyeAIApp().mediaFrameAnalyzer
        ) {
            CoroutineScope(Dispatchers.Main).launch {
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "onStreamingComplete CALLBACK: Fired, but logic is now handled by the global callback."
                )
            }
        }

        SpeechManager.stream = stateMachine.getStreamingHandler()

        eyeAIApp().currentStateMachine = stateMachine

        val update = when (currentState) {
            MainActivity.State.IDLE -> stateMachine.handleIdle(final)
            MainActivity.State.SETTINGS_MENU -> stateMachine.handleSettingsMenu(final)
            MainActivity.State.SETTINGS_CHOICE -> stateMachine.handleSettingsChoice(final)
            MainActivity.State.SETTINGS_ACTION -> stateMachine.handleSettingsAction(final)
        }

        // Logging der state transition
        Log.d(EyeAIApp.APP_LOG_TAG, "State transition: $currentState -> ${update.newState}")
        currentState = update.newState
        eyeAIApp().lastLlmJsonResponse = update.newJson
    }

    fun updateLlmResponseText(text: String) {
        _uiState.update { it.copy(llmResponseText = text) }
    }

    private fun hasPermission(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(
            eyeAIApp(),
            permission
        ) == PackageManager.PERMISSION_GRANTED
    }

    @RequiresApi(Build.VERSION_CODES.P)
    fun onResume() {
        if (eyeAIApp().settings.enableSpeechRecognition && !hasPermission(Manifest.permission.RECORD_AUDIO)) {
            _uiState.update { it.copy(appMissingVoskPermission = true) }
        }
        updateVoskStatusText()
        val isLLMConfigured = eyeAIApp().settings.googleAiStudioApiKey?.isEmpty() == false
        updateLlmResponseText(
            if (isLLMConfigured)
                ""
            else
                eyeAIApp().getString(R.string.setup_llm_notice)
        )
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun initCamera(cameraPreviewView: PreviewView?, lifecycleOwner: LifecycleOwner) {
        if (eyeAIApp().settings.inputSource == eyeAIApp().getString(R.string.input_is_camera)) {
            if (hasPermission(Manifest.permission.CAMERA)) {
                //ungrantedPermissionsNotice!!.visibility = GONE

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
                //ungrantedPermissionsNotice!!.visibility = VISIBLE
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
                //TODO implement Dialog
                Log.d(
                    LOG_TAG,
                    "[MainViewModel.initCamera] Input Source is media but selected file is empty"
                )
                /*
                val builder = AlertDialog.Builder(eyeAIApp())
                builder.setMessage("No media file has been selected. Please select one in the settings menu")
                    .setPositiveButton("Open settings") { dialog, id ->
                        startActivity(Intent(this, SettingsActivity::class.java))
                        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                    }
                builder.create().show()
                 */
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

                        if (!voskUserStart.get()) {
                            startVoskListening()
                        }


                    },
                    onDoubleClick = {
                        Log.i("CLICK", "DOUBLE")

                        State.IDLE

                        if (voskUserStart.get()) {
                            SpeechManager.forceStop()
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

    fun eyeAIApp(): EyeAIApp {
        return getApplication()
    }
}