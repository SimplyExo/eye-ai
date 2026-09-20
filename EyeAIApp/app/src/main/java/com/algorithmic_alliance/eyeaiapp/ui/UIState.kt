package com.algorithmic_alliance.eyeaiapp.ui

import android.graphics.Bitmap
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import uniffi.NativeLib.UniffiDetectedObject

data class UIState(
	val voskListening: Boolean = false,
	val batteryOptimizationIgnored: Boolean = false,
	val ttsSpeaking: Boolean = false,
	val actionStartedFromSettings: Boolean = false,
	val settingsOpened: Boolean = false,
	val reloadSettingsPageKey: Int = 0,
	val reloadDebugPageKey: Int = 0,
	val appMissingSelectedMediaSource: Boolean = false,
	val appMissingVoskPermission: Boolean = false,
	val appMissingCameraPermission: Boolean = false,
	val appMissingVisionPermission: Boolean = false,
	val appNotExemptFromBatteryOptimization: Boolean = false,
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
	val segmentationOverlayEnabled: Boolean = true,
) {
	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as UIState

		if (voskListening != other.voskListening) return false
		if (ttsSpeaking != other.ttsSpeaking) return false
		if (actionStartedFromSettings != other.actionStartedFromSettings) return false
		if (settingsOpened != other.settingsOpened) return false
		if (reloadSettingsPageKey != other.reloadSettingsPageKey) return false
		if (reloadDebugPageKey != other.reloadDebugPageKey) return false
		if (appMissingSelectedMediaSource != other.appMissingSelectedMediaSource) return false
		if (appMissingVoskPermission != other.appMissingVoskPermission) return false
		if (appMissingCameraPermission != other.appMissingCameraPermission) return false
		if (appMissingVisionPermission != other.appMissingVisionPermission) return false
		if (appNotExemptFromBatteryOptimization != other.appNotExemptFromBatteryOptimization) return false
		if (permissionTutorialCompleted != other.permissionTutorialCompleted) return false
		if (connectionTutorialCompleted != other.connectionTutorialCompleted) return false
		if (segmentationOverlayEnabled != other.segmentationOverlayEnabled) return false
		if (batteryOptimizationIgnored != other.batteryOptimizationIgnored) return false
		if (speechRecognitionFinalResultText != other.speechRecognitionFinalResultText) return false
		if (speechRecognitionPartialResultText != other.speechRecognitionPartialResultText) return false
		if (speechResponseText != other.speechResponseText) return false
		if (depthPreviewBitmap != other.depthPreviewBitmap) return false
		if (mediaPreviewBitmap != other.mediaPreviewBitmap) return false
		if (debugInputPreviewBitmap != other.debugInputPreviewBitmap) return false
		if (performanceText != other.performanceText) return false
		if (!detectedObjects.contentEquals(other.detectedObjects)) return false
		if (debugSegmentationBitmap != other.debugSegmentationBitmap) return false
		if (cameraResolution != other.cameraResolution) return false
		if (!ocrResults.contentEquals(other.ocrResults)) return false

		return true
	}

	override fun hashCode(): Int {
		var result = voskListening.hashCode()
		result = 31 * result + ttsSpeaking.hashCode()
		result = 31 * result + actionStartedFromSettings.hashCode()
		result = 31 * result + settingsOpened.hashCode()
		result = 31 * result + reloadSettingsPageKey
		result = 31 * result + reloadDebugPageKey
		result = 31 * result + appMissingSelectedMediaSource.hashCode()
		result = 31 * result + appMissingVoskPermission.hashCode()
		result = 31 * result + appMissingCameraPermission.hashCode()
		result = 31 * result + appMissingVisionPermission.hashCode()
		result = 31 * result + appNotExemptFromBatteryOptimization.hashCode()
		result = 31 * result + permissionTutorialCompleted.hashCode()
		result = 31 * result + connectionTutorialCompleted.hashCode()
		result = 31 * result + segmentationOverlayEnabled.hashCode()
		result = 31 * result + batteryOptimizationIgnored.hashCode()
		result = 31 * result + speechRecognitionFinalResultText.hashCode()
		result = 31 * result + speechRecognitionPartialResultText.hashCode()
		result = 31 * result + speechResponseText.hashCode()
		result = 31 * result + (depthPreviewBitmap?.hashCode() ?: 0)
		result = 31 * result + (mediaPreviewBitmap?.hashCode() ?: 0)
		result = 31 * result + (debugInputPreviewBitmap?.hashCode() ?: 0)
		result = 31 * result + performanceText.hashCode()
		result = 31 * result + detectedObjects.contentHashCode()
		result = 31 * result + (debugSegmentationBitmap?.hashCode() ?: 0)
		result = 31 * result + cameraResolution.hashCode()
		result = 31 * result + ocrResults.contentHashCode()
		return result
	}
}