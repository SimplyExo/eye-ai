package com.algorithmic_alliance.eyeaiapp.UI

import android.Manifest
import android.app.Application
import android.content.pm.PackageManager
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntimeService
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Activity-facing projection of [com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntime].
 * It owns only transient UI state and never starts/stops resources from an
 * Activity lifecycle callback.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as EyeAIApp
    private val runtime = app.runtime

    private val _uiState = MutableStateFlow(UIState())
    val uiState: StateFlow<UIState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            runtime.state.collect { runtimeState ->
                _uiState.update {
                    it.copy(
                        voskListening = runtimeState.voskListening,
                        ttsSpeaking = runtimeState.ttsSpeaking,
                        speechRecognitionFinalResultText =
                            runtimeState.speechRecognitionFinalResultText,
                        speechRecognitionPartialResultText =
                            runtimeState.speechRecognitionPartialResultText,
                        speechResponseText = runtimeState.speechResponseText,
                        depthPreviewBitmap = runtimeState.depthPreviewBitmap,
                        debugInputPreviewBitmap = runtimeState.debugInputPreviewBitmap,
                        mediaPreviewBitmap = runtimeState.mediaPreviewBitmap,
                        performanceText = runtimeState.performanceText,
                        detectedObjects = runtimeState.detectedObjects,
                        cameraResolution = runtimeState.cameraResolution,
                        ocrResults = runtimeState.ocrResults,
                    )
                }
            }
        }
    }

    fun onEvent(event: UIEvent) {
        when (event) {
            UIEvent.VoskListeningChanged -> {
                Log.d(LOG_TAG, "[MainViewModel] VoskListeningChanged")
                runtime.toggleListening()
            }
            UIEvent.UpdateVoskStatusText -> runtime.updateVoskStatusText()
            UIEvent.OnReloadSettingsPage -> reloadSettingsPage()
            UIEvent.InitVoskService -> runtime.initSpeechService()
            UIEvent.CloseVoskService -> runtime.closeSpeechService()
            UIEvent.OnReloadDebugPage -> reloadDebugPage()
            UIEvent.UpdateSettings -> app.updateSettings()
            is UIEvent.OnUpdatePermissionTutorialCompleted ->
                _uiState.update { it.copy(permissionTutorialCompleted = event.value) }
            is UIEvent.OnUpdateConnectionTutorialCompleted ->
                _uiState.update { it.copy(connectionTutorialCompleted = event.value) }
            UIEvent.UpdateSpeechStatusText -> runtime.updateVoskStatusText()
            UIEvent.OnOpenSettings -> onOpenSettings()
            UIEvent.OnReturnFromSettings -> onReturnFromSettings()
            is UIEvent.OnUpdateSettingsOpened ->
                _uiState.update { it.copy(settingsOpened = event.value) }
            is UIEvent.OnUpdateAppMissingCameraPermission ->
                _uiState.update { it.copy(appMissingCameraPermission = event.value) }
            is UIEvent.OnUpdateAppMissingVoskPermission ->
                _uiState.update { it.copy(appMissingVoskPermission = event.value) }
            is UIEvent.UIinitCamera -> initCamera(event.previewView)
            is UIEvent.UIDetachCameraPreview -> runtime.detachPreview(event.previewView)
            is UIEvent.OnUpdateActionStartedFromSettings ->
                _uiState.update { it.copy(actionStartedFromSettings = event.value) }
            is UIEvent.OnUpdateTTSSpeaking ->
                _uiState.update { it.copy(ttsSpeaking = event.value) }
            is UIEvent.OnUpdateAppMissingSelectedMediaSource ->
                _uiState.update { it.copy(appMissingSelectedMediaSource = event.value) }
        }
    }

    /** Refreshes UI permission/status data; it intentionally leaves runtime resources alone. */
    fun onResume() {
        Log.d(LOG_TAG, "[MainViewModel] OnResume: refreshing UI projection")
        if (!_uiState.value.actionStartedFromSettings && !_uiState.value.settingsOpened) {
            reloadDebugPage()
        }
        if (
            !hasPermission(Manifest.permission.CAMERA) &&
            _uiState.value.permissionTutorialCompleted
        ) {
            _uiState.update { it.copy(appMissingCameraPermission = true) }
        }
        if (
            app.settings.enableSpeechRecognition &&
            !hasPermission(Manifest.permission.RECORD_AUDIO) &&
            _uiState.value.permissionTutorialCompleted
        ) {
            _uiState.update { it.copy(appMissingVoskPermission = true) }
        }
        runtime.updateVoskStatusText()
    }

    /** Kept for callers that used the old ViewModel API; no lifecycle shutdown occurs here. */
    fun onPause() = Unit

    fun setTTSSpeaking(value: Boolean) {
        _uiState.update { it.copy(ttsSpeaking = value) }
    }

    fun setVoskListening(value: Boolean) {
        _uiState.update { it.copy(voskListening = value) }
    }

    fun updateSpeechResponseText(text: String) {
        _uiState.update { it.copy(speechResponseText = text) }
    }

    fun updateSettings() = app.updateSettings()

    fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

    fun eyeAIApp(): EyeAIApp = app

    private fun onOpenSettings() {
        // Opening a UI destination must not interrupt the service-owned
        // analysis pipeline. Settings updates are applied independently via
        // EyeAIApp.updateSettings().
        _uiState.update { it.copy(settingsOpened = true) }
    }

    private fun onReturnFromSettings() {
        _uiState.update {
            it.copy(
                settingsOpened = false,
                detectedObjects = emptyArray(),
                ocrResults = emptyArray(),
            )
        }
        // Do not start a service from a composable's disposal: disposal also
        // happens while the Activity/task is being destroyed. The returning
        // Home/Debug destination attaches to the already active runtime (or
        // starts a source changed in settings) through UIinitCamera.
    }

    private fun initCamera(previewView: androidx.camera.view.PreviewView?) {
        _uiState.update {
            it.copy(
                detectedObjects = emptyArray(),
                ocrResults = emptyArray(),
            )
        }
        if (app.settings.inputSource == app.getString(R.string.input_is_camera)) {
            if (!hasPermission(Manifest.permission.CAMERA)) {
                _uiState.update { it.copy(appMissingCameraPermission = true) }
                return
            }
            // This call is made by a visible Compose destination. The service
            // then becomes the CameraX LifecycleOwner and survives screen-off.
            app.runtime.attachPreview(previewView)
            startRuntimeIfCameraPermissionGranted()
        } else if (
            app.settings.inputSource == app.getString(R.string.input_is_media) &&
            app.settings.mediaSource.isNullOrEmpty()
        ) {
            _uiState.update { it.copy(appMissingSelectedMediaSource = true) }
        } else if (
            app.settings.inputSource == app.getString(R.string.input_is_media) ||
                app.settings.inputSource == app.getString(R.string.input_is_eyeaivision)
        ) {
            // These existing non-camera sources are still created by the
            // runtime. If speech is enabled, the same FGS also has its
            // microphone type; otherwise the current camera mode remains the
            // only continuous FGS source in this task.
            if (hasPermission(Manifest.permission.RECORD_AUDIO)) {
                EyeAIRuntimeService.startFromVisible(app)
            }
        }
    }

    private fun startRuntimeIfCameraPermissionGranted() {
        if (app.settings.inputSource != app.getString(R.string.input_is_camera)) {
            if (hasPermission(Manifest.permission.RECORD_AUDIO)) {
                EyeAIRuntimeService.startFromVisible(app)
            }
            return
        }
        if (!hasPermission(Manifest.permission.CAMERA)) {
            _uiState.update { it.copy(appMissingCameraPermission = true) }
            return
        }
        EyeAIRuntimeService.startFromVisible(app)
    }

    private fun hasPermission(permission: String): Boolean = ContextCompat.checkSelfPermission(
        app,
        permission,
    ) == PackageManager.PERMISSION_GRANTED

    private fun reloadDebugPage() {
        _uiState.update { it.copy(reloadDebugPageKey = it.reloadDebugPageKey + 1) }
    }

    private fun reloadSettingsPage() {
        _uiState.update { it.copy(reloadSettingsPageKey = it.reloadSettingsPageKey + 1) }
    }
}
