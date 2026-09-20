package com.algorithmic_alliance.eyeaiapp.runtime

import android.graphics.Bitmap
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.camera.FrameAnalysisUpdate
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import uniffi.NativeLib.UniffiDetectedObject

/** Immutable state exposed to short-lived UI observers. */
data class EyeAIRuntimeState(
	val operationActive: Boolean = false,
	val cameraActive: Boolean = false,
	val voskListening: Boolean = false,
	val ttsSpeaking: Boolean = false,
	val speechRecognitionFinalResultText: String = "",
	val speechRecognitionPartialResultText: String = "",
	val speechResponseText: String = "",
	val depthPreviewBitmap: Bitmap? = null,
	val segmentationPreviewBitmap: Bitmap? = null,
	val debugInputPreviewBitmap: Bitmap? = null,
	val debugSegmentationBitmap: Bitmap? = null,
	val mediaPreviewBitmap: Bitmap? = null,
	val performanceText: String = "",
	val detectedObjects: Array<UniffiDetectedObject> = emptyArray(),
	val cameraResolution: Size = Size(720, 1280),
	val ocrResults: Array<TextBoundingBox> = emptyArray(),
	val lastError: String? = null,
) {
	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as EyeAIRuntimeState

		if (operationActive != other.operationActive) return false
		if (cameraActive != other.cameraActive) return false
		if (voskListening != other.voskListening) return false
		if (ttsSpeaking != other.ttsSpeaking) return false
		if (speechRecognitionFinalResultText != other.speechRecognitionFinalResultText) return false
		if (speechRecognitionPartialResultText != other.speechRecognitionPartialResultText) return false
		if (speechResponseText != other.speechResponseText) return false
		if (depthPreviewBitmap != other.depthPreviewBitmap) return false
		if (segmentationPreviewBitmap != other.segmentationPreviewBitmap) return false
		if (debugInputPreviewBitmap != other.debugInputPreviewBitmap) return false
		if (debugSegmentationBitmap != other.debugSegmentationBitmap) return false
		if (mediaPreviewBitmap != other.mediaPreviewBitmap) return false
		if (performanceText != other.performanceText) return false
		if (!detectedObjects.contentEquals(other.detectedObjects)) return false
		if (cameraResolution != other.cameraResolution) return false
		if (!ocrResults.contentEquals(other.ocrResults)) return false
		if (lastError != other.lastError) return false

		return true
	}

	override fun hashCode(): Int {
		var result = operationActive.hashCode()
		result = 31 * result + cameraActive.hashCode()
		result = 31 * result + voskListening.hashCode()
		result = 31 * result + ttsSpeaking.hashCode()
		result = 31 * result + speechRecognitionFinalResultText.hashCode()
		result = 31 * result + speechRecognitionPartialResultText.hashCode()
		result = 31 * result + speechResponseText.hashCode()
		result = 31 * result + (depthPreviewBitmap?.hashCode() ?: 0)
		result = 31 * result + (segmentationPreviewBitmap?.hashCode() ?: 0)
		result = 31 * result + (debugInputPreviewBitmap?.hashCode() ?: 0)
		result = 31 * result + (debugSegmentationBitmap?.hashCode() ?: 0)
		result = 31 * result + (mediaPreviewBitmap?.hashCode() ?: 0)
		result = 31 * result + performanceText.hashCode()
		result = 31 * result + detectedObjects.contentHashCode()
		result = 31 * result + cameraResolution.hashCode()
		result = 31 * result + ocrResults.contentHashCode()
		result = 31 * result + (lastError?.hashCode() ?: 0)
		return result
	}
}

internal fun EyeAIRuntimeState.withAnalysis(update: FrameAnalysisUpdate): EyeAIRuntimeState = copy(
	depthPreviewBitmap = update.depthPreviewBitmap ?: depthPreviewBitmap,
	segmentationPreviewBitmap = update.debugSegmentationBitmap ?: segmentationPreviewBitmap,
	debugInputPreviewBitmap = update.debugInputBitmap ?: debugInputPreviewBitmap,
	debugSegmentationBitmap = update.debugSegmentationBitmap ?: debugSegmentationBitmap,
	performanceText = update.performanceText ?: performanceText,
	detectedObjects = update.detectedObjects ?: detectedObjects,
	cameraResolution = update.frameSize ?: cameraResolution,
)
