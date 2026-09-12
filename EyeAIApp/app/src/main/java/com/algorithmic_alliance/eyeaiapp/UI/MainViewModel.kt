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
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.sample
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.time.Duration.Companion.seconds
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG

/**
 * Activity-facing projection of [com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntime].
 * It owns only transient UI state and never starts/stops resources from an
 * Activity lifecycle callback.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {
	private val app = application as EyeAIApp
	private val runtime = app.runtime

	fun isSpeechRecognitionEnabled(): Boolean = eyeAIApp().settings.enableSpeechRecognition

	fun appInputSource(): String = eyeAIApp().settings.inputSource as String

	private val _uiState = MutableStateFlow(UIState())
	val uiState: StateFlow<UIState> = _uiState.asStateFlow()

	val connectionPageUIState: StateFlow<ConnectionPageUIState> = _uiState.map {
		ConnectionPageUIState(
			visionPermissionsNotGranted = it.appMissingVisionPermission,
		)
	}.distinctUntilChanged()
		.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), ConnectionPageUIState())
	val chooseConnectionPageUIState: StateFlow<ChooseConnectionPageUIState> = _uiState.map {
		ChooseConnectionPageUIState(
			connectionTutorialCompleted = it.connectionTutorialCompleted,
		)
	}.distinctUntilChanged().stateIn(
		viewModelScope, SharingStarted.WhileSubscribed(5000), ChooseConnectionPageUIState()
	)

	val homePageUIState: StateFlow<HomePageUIState> = _uiState.map {
		HomePageUIState(
			voskListening = it.voskListening,
			ttsSpeaking = it.ttsSpeaking,
		)
	}.distinctUntilChanged()
		.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), HomePageUIState())

	@OptIn(FlowPreview::class)
	val performanceStatusCardUIState: StateFlow<PerformanceStatusCardUIState> =
		_uiState.sample(10.seconds).map {
			PerformanceStatusCardUIState(
				performanceText = it.performanceText,
			)
		}.distinctUntilChanged().stateIn(
			viewModelScope, SharingStarted.WhileSubscribed(5000), PerformanceStatusCardUIState()
		)

	val voskStatusCardUIState: StateFlow<VoskStatusCardUIState> = _uiState.map {
		VoskStatusCardUIState(
			ttsSpeaking = it.ttsSpeaking,
			speechRecognitionFinalResultText = it.speechRecognitionFinalResultText
		)
	}.distinctUntilChanged()
		.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), VoskStatusCardUIState())

	val settingsPageUIState: StateFlow<SettingsPageUIState> = _uiState.map {
		SettingsPageUIState(
			reloadSettingsPageKey = it.reloadSettingsPageKey,
		)
	}.distinctUntilChanged()
		.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), SettingsPageUIState())

	val selectSettingUIState: StateFlow<SelectSettingUIState> = _uiState.map {
		SelectSettingUIState(
			visionPermissionsNotGranted = it.appMissingVisionPermission,
			reloadSettingsPageKey = it.reloadSettingsPageKey
		)
	}.distinctUntilChanged()
		.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), SelectSettingUIState())
	val debugPageUIState: StateFlow<DebugPageUIState> = _uiState.map {
		DebugPageUIState(
			reloadDebugPageKey = it.reloadDebugPageKey,
			voskListening = it.voskListening,
			ttsSpeaking = it.ttsSpeaking,
			speechRecognitionFinalResultText = it.speechRecognitionFinalResultText,
			speechRecognitionPartialResultText = it.speechRecognitionPartialResultText,
			speechResponseText = it.speechResponseText
		)
	}.distinctUntilChanged()
		.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), DebugPageUIState())

	val mediaPreviewUIState: StateFlow<MediaPreviewUIState> = _uiState.map {
		MediaPreviewUIState(
			mediaPreviewBitmap = it.mediaPreviewBitmap
		)
	}.distinctUntilChanged()
		.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), MediaPreviewUIState())
	val debugInputBitmapPreviewUIState: StateFlow<DebugInputBitmapPreviewUIState> = _uiState.map {
		DebugInputBitmapPreviewUIState(
			debugInputPreviewBitmap = it.debugInputPreviewBitmap
		)
	}.distinctUntilChanged().stateIn(
		viewModelScope, SharingStarted.WhileSubscribed(5000), DebugInputBitmapPreviewUIState()
	)

	val objectDetectionOverlayUIState: StateFlow<ObjectDetectionOverlayUIState> = _uiState.map {
		ObjectDetectionOverlayUIState(
			detectedObjects = it.detectedObjects, cameraResolution = it.cameraResolution
		)
	}.distinctUntilChanged().stateIn(
		viewModelScope, SharingStarted.WhileSubscribed(5000), ObjectDetectionOverlayUIState()
	)

	val depthOverlayUIState: StateFlow<DepthOverlayUIState> = _uiState.map {
		DepthOverlayUIState(
			depthPreviewBitmap = it.depthPreviewBitmap, performanceText = it.performanceText
		)
	}.distinctUntilChanged()
		.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), DepthOverlayUIState())

	val uiDialogsUIState: StateFlow<UIDialogsUIState> = _uiState.map {
		UIDialogsUIState(
			appMissingSelectedMediaSource = it.appMissingSelectedMediaSource,
			appMissingVoskPermission = it.appMissingVoskPermission,
			appMissingCameraPermission = it.appMissingCameraPermission,
			appMissingVisionPermission = it.appMissingVisionPermission
		)
	}.distinctUntilChanged()
		.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), UIDialogsUIState())

	init {
		viewModelScope.launch {
			runtime.state.collect { runtimeState ->
				_uiState.update {
					it.copy(
						voskListening = runtimeState.voskListening,
						ttsSpeaking = runtimeState.ttsSpeaking,
						speechRecognitionFinalResultText = runtimeState.speechRecognitionFinalResultText,
						speechRecognitionPartialResultText = runtimeState.speechRecognitionPartialResultText,
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
        Log.i(LOG_TAG, "!!! MainViewModel onEvent: $event !!!")
        when (event) {
            UIEvent.VoskListeningChanged -> {
                Log.d(LOG_TAG, "[MainViewModel] VoskListeningChanged")
                runtime.toggleListening()
                runtime.updateVoskStatusText()
            }

			UIEvent.UpdateVoskStatusText -> runtime.updateVoskStatusText()
			UIEvent.OnReloadSettingsPage -> reloadSettingsPage()
			UIEvent.InitVoskService -> runtime.initSpeechService()
			UIEvent.CloseVoskService -> runtime.closeSpeechService()
			UIEvent.OnReloadDebugPage -> reloadDebugPage()
			UIEvent.UpdateSettings -> app.updateSettings()
			is UIEvent.OnUpdatePermissionTutorialCompleted -> _uiState.update {
				it.copy(
					permissionTutorialCompleted = event.value
				)
			}

			is UIEvent.OnUpdateConnectionTutorialCompleted -> _uiState.update {
				it.copy(
					connectionTutorialCompleted = event.value
				)
			}

			UIEvent.UpdateSpeechStatusText -> runtime.updateVoskStatusText()
			UIEvent.OnOpenSettings -> onOpenSettings()
			UIEvent.OnReturnFromSettings -> onReturnFromSettings()
			is UIEvent.OnUpdateSettingsOpened -> _uiState.update { it.copy(settingsOpened = event.value) }

			is UIEvent.OnUpdateAppMissingCameraPermission -> _uiState.update {
				it.copy(
					appMissingCameraPermission = event.value
				)
			}

			is UIEvent.OnUpdateAppMissingVoskPermission -> _uiState.update {
				it.copy(
					appMissingVoskPermission = event.value
				)
			}

            is UIEvent.UIinitCamera -> {
                Log.d(LOG_TAG, "[MainViewModel] UIinitCamera received")
                initCamera(event.previewView)
            }
            is UIEvent.UIDetachCameraPreview -> runtime.detachPreview(event.previewView)
            is UIEvent.OnUpdateActionStartedFromSettings -> _uiState.update {
                it.copy(
                    actionStartedFromSettings = event.value
                )
            }

			is UIEvent.OnUpdateTTSSpeaking -> _uiState.update { it.copy(ttsSpeaking = event.value) }

			is UIEvent.OnUpdateAppMissingSelectedMediaSource -> _uiState.update {
				it.copy(
					appMissingSelectedMediaSource = event.value
				)
			}

			is UIEvent.OnUpdateAppMissingVisionPermission -> _uiState.update {
				it.copy(
					appMissingVisionPermission = event.value
				)
			}
		}
	}

	/** Refreshes UI permission/status data; it intentionally leaves runtime resources alone. */
	fun onResume() {
		Log.d(LOG_TAG, "[MainViewModel] OnResume: refreshing UI projection")
		if (!_uiState.value.actionStartedFromSettings && !_uiState.value.settingsOpened) {
			reloadDebugPage()
		}
		// permission checks after onResume are now handled bei EyeAIUI.kt
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
        val currentSource = app.settings.inputSource
        Log.d(LOG_TAG, "[MainViewModel] initCamera: source=$currentSource")
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
        } else if (app.settings.inputSource == app.getString(R.string.input_is_media) && app.settings.mediaSource.isNullOrEmpty()) {
            _uiState.update { it.copy(appMissingSelectedMediaSource = true) }
        } else if (app.settings.inputSource == app.getString(R.string.input_is_media) || app.settings.inputSource == app.getString(
                R.string.input_is_eyeaivision
            )
        ) {
            // Start the service for EyeAIVision or Media to ensure background continuity.
            EyeAIRuntimeService.startFromVisible(app)
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
