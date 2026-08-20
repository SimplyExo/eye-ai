package com.algorithmic_alliance.eyeaiapp.UI

import android.app.Application
import android.os.Build
import android.os.Looper
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.MainActivity
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.PermissionManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG
import com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio.SpeechManager
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateMachine
import com.algorithmic_alliance.eyeaiapp.vibrate
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
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
                State.IDLE
                if (!voskUserStart.get()) {
                    startVosk()
                } else {
                    stopVosk()
                }
            }

            UIEvent.UpdateVoskStatusText -> {

            }

            UIEvent.InitVoskService -> {
                Log.d(LOG_TAG, "[MainViewModel] InitVoskService")
                initVoskService()
            }

            UIEvent.CloseVoskService -> {

            }
        }
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

        uniffi.NativeLib.setObjectAudioPaused(false)
        uniffi.NativeLib.setDepthAudioPaused(false)
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
    private fun updateVoskStatusText() {
        _uiState.update {
            it.copy(
                speechRecognitionFinalResultText = when {
                    !eyeAIApp().settings.enableSpeechRecognition -> "Spracherkennung deaktiviert"
                    voskUserStart.get() -> eyeAIApp().getString(R.string.speech_recognition_ready)
                    else -> "Vosk bereit - Button klicken zum Starten"
                }
            )
        }
        /*
        speechRecognitionFinalResultText?.text = when {
            !permissionManager.isMicrophonePermissionGranted() ->
                "Mikrofon-Berechtigung erforderlich"

            !eyeAIApp().settings.enableSpeechRecognition ->
                "Spracherkennung deaktiviert"

            voskUserStart.get() ->
                getString(R.string.speech_recognition_ready)

            else ->
                "Vosk bereit - Button klicken zum Starten"
        }
         */


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
            { text -> _uiState.update { it.copy(llmResponseText =  text) }},
            {text -> _uiState.update { it.copy(llmResponseText = it.llmResponseText + text ) }},
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

    fun updateLlmResponseText(text: String){
        _uiState.update { it.copy(llmResponseText = text) }
    }

    fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

    private fun eyeAIApp(): EyeAIApp {
        return getApplication()
    }
}