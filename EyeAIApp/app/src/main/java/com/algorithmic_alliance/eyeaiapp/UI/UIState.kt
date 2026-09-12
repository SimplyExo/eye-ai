package com.algorithmic_alliance.eyeaiapp.UI

import android.graphics.Bitmap
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import uniffi.NativeLib.UniffiDetectedObject

data class UIState(
	val voskListening: Boolean = false,
	val ttsSpeaking: Boolean = false,
	val actionStartedFromSettings: Boolean = false,
	val settingsOpened: Boolean = false,
	val reloadSettingsPageKey: Int = 0,
	val reloadDebugPageKey: Int = 0,
	val appMissingSelectedMediaSource: Boolean = false,
	val appMissingVoskPermission: Boolean = false,
	val appMissingCameraPermission: Boolean = false,
	val appMissingVisionPermission: Boolean = false,
	val permissionTutorialCompleted: Boolean = false,
	val connectionTutorialCompleted: Boolean = false,
	val speechRecognitionFinalResultText: String = "",
	val speechRecognitionPartialResultText: String = "",
	val speechResponseText: String = "",
	val depthPreviewBitmap: Bitmap? = null,
	val mediaPreviewBitmap: Bitmap? = null,
	val debugInputPreviewBitmap: Bitmap? = null,
	val performanceText: String = "",
	val detectedObjects: Array<UniffiDetectedObject> = emptyArray(),
	val debugSegmentationBitmap: Bitmap? = null,
	val cameraResolution: Size = Size(720, 1280),
	val ocrResults: Array<TextBoundingBox> = emptyArray(),
	val segmentationOverlayEnabled: Boolean = true
)