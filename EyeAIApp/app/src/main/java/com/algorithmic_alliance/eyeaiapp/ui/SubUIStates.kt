package com.algorithmic_alliance.eyeaiapp.ui

import android.graphics.Bitmap
import android.util.Size
import androidx.compose.runtime.Immutable
import uniffi.NativeLib.UniffiDetectedObject


@Immutable
data class ConnectionPageUIState(
	val visionPermissionsNotGranted: Boolean = false,
)

@Immutable
data class ChooseConnectionPageUIState(
	val connectionTutorialCompleted: Boolean = false
)

@Immutable
data class HomePageUIState(
	val voskListening: Boolean = false, val ttsSpeaking: Boolean = false
)

@Immutable
data class PerformanceStatusCardUIState(
	val performanceText: String = ""
)

@Immutable
data class VoskStatusCardUIState(
	val ttsSpeaking: Boolean = false, val speechRecognitionFinalResultText: String = ""
)

@Immutable
data class SettingsPageUIState(
	val reloadSettingsPageKey: Int = 0
)

@Immutable
data class SelectSettingUIState(
	val visionPermissionsNotGranted: Boolean = false, val reloadSettingsPageKey: Int = 0
)

@Immutable
data class DebugPageUIState(
	val reloadDebugPageKey: Int = 0,
	val voskListening: Boolean = false,
	val ttsSpeaking: Boolean = false,
	val speechRecognitionFinalResultText: String = "",
	val speechRecognitionPartialResultText: String = "",
	val speechResponseText: String = "",
	val segmentationOverlayEnabled: Boolean = true,
)

@Immutable
data class MediaPreviewUIState(
	val mediaPreviewBitmap: Bitmap? = null
)

@Immutable
data class DebugInputBitmapPreviewUIState(
	val debugInputPreviewBitmap: Bitmap? = null
)

@Immutable
data class ObjectDetectionOverlayUIState(
	val detectedObjects: Array<UniffiDetectedObject> = emptyArray(),
	val cameraResolution: Size = Size(720, 1280),
) {
	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as ObjectDetectionOverlayUIState

		if (!detectedObjects.contentEquals(other.detectedObjects)) return false
		if (cameraResolution != other.cameraResolution) return false

		return true
	}

	override fun hashCode(): Int {
		var result = detectedObjects.contentHashCode()
		result = 31 * result + cameraResolution.hashCode()
		return result
	}
}

@Immutable
data class DepthOverlayUIState(
	val depthPreviewBitmap: Bitmap? = null,
	val performanceText: String = "",
)

@Immutable
data class SegmentationOverlayUIState(
	val debugSegmentationBitmap: Bitmap? = null,
	val segmentationOverlayEnabled: Boolean = true
)

@Immutable
data class UIDialogsUIState(
	val appMissingSelectedMediaSource: Boolean = false,
	val appMissingVoskPermission: Boolean = false,
	val appMissingCameraPermission: Boolean = false,
	val appMissingVisionPermission: Boolean = false,
	val appNotExemptFormBatteryOptimization: Boolean = false,
)
