package com.algorithmic_alliance.eyeaiapp.camera

import android.hardware.camera2.CameraCharacteristics
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import com.algorithmic_alliance.eyeaiapp.rel2abs.Rel2AbsCameraIntrinsics

data class CameraCalibration(
	private val fxPx: Float,
	private val fyPx: Float,
	private val referenceWidthPx: Int,
	private val referenceHeightPx: Int,
) {
	fun forFrame(widthPx: Int, heightPx: Int, rotationDegrees: Int): Rel2AbsCameraIntrinsics {
		val quarterTurn = rotationDegrees % 180 != 0
		val fx = if (quarterTurn) {
			fyPx * widthPx / referenceHeightPx.toFloat()
		} else {
			fxPx * widthPx / referenceWidthPx.toFloat()
		}
		val fy = if (quarterTurn) {
			fxPx * heightPx / referenceWidthPx.toFloat()
		} else {
			fyPx * heightPx / referenceHeightPx.toFloat()
		}
		return Rel2AbsCameraIntrinsics(fx, fy)
	}
}

data class CameraSelection(
	val cameraSelector: CameraSelector,
	val calibration: CameraCalibration?,
)

/**
 * @return Selector that selects the most wide angle sens back camera
 */
@OptIn(ExperimentalCamera2Interop::class)
fun mostWideCameraSelection(cameraProvider: ProcessCameraProvider): CameraSelection {
	var widestCamera: CameraInfo? = null
	var smallestFocalLength: Float? = null

	for (cameraInfo in cameraProvider.availableCameraInfos) {
		if (cameraInfo.lensFacing != CameraSelector.LENS_FACING_BACK) {
			continue
		}

		val camera2CameraInfo = Camera2CameraInfo.from(cameraInfo)
		val focalLengths = camera2CameraInfo.getCameraCharacteristic(
			CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS
		)

		if (focalLengths != null && focalLengths.isNotEmpty()) {
			// focalLengths in ascending order: smallest at first
			val focalLength = focalLengths[0]

			if (smallestFocalLength == null || focalLength <= smallestFocalLength) {
				smallestFocalLength = focalLength
				widestCamera = cameraInfo
			}
		}
	}

	val selector = widestCamera?.cameraSelector ?: CameraSelector.DEFAULT_BACK_CAMERA
	return CameraSelection(selector, widestCamera?.let(::cameraCalibration))
}

@OptIn(ExperimentalCamera2Interop::class)
fun mostWideCameraSelector(cameraProvider: ProcessCameraProvider): CameraSelector =
	mostWideCameraSelection(cameraProvider).cameraSelector

@OptIn(ExperimentalCamera2Interop::class)
private fun cameraCalibration(cameraInfo: CameraInfo): CameraCalibration? {
	val camera2Info = Camera2CameraInfo.from(cameraInfo)
	val activeArray = camera2Info.getCameraCharacteristic(
		CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE,
	) ?: return null
	val referenceWidth = activeArray.width()
	val referenceHeight = activeArray.height()
	if (referenceWidth <= 0 || referenceHeight <= 0) return null

	val intrinsic = camera2Info.getCameraCharacteristic(
		CameraCharacteristics.LENS_INTRINSIC_CALIBRATION,
	)
	val focalLengths = camera2Info.getCameraCharacteristic(
		CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS,
	)
	val physicalSize = camera2Info.getCameraCharacteristic(
		CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE,
	)
	val focalPair = if (intrinsic != null && intrinsic.size >= 2 && intrinsic[0] > 0f && intrinsic[1] > 0f) {
		intrinsic[0] to intrinsic[1]
	} else if (
		focalLengths != null && focalLengths.isNotEmpty() && focalLengths[0] > 0f &&
		physicalSize != null && physicalSize.width > 0f && physicalSize.height > 0f
	) {
		(focalLengths[0] / physicalSize.width * referenceWidth) to
			(focalLengths[0] / physicalSize.height * referenceHeight)
	} else {
		null
	}
	return focalPair?.let { (fx, fy) ->
		CameraCalibration(fx, fy, referenceWidth, referenceHeight)
	}
}

/**
 * @return Selector that chooses the smallest possible resolution that still fits the [inputSize]
 */
fun performanceResolutionSelector(inputSize: Size): ResolutionSelector {
	return ResolutionSelector.Builder()
		.setAllowedResolutionMode(ResolutionSelector.PREFER_CAPTURE_RATE_OVER_HIGHER_RESOLUTION)
		.setResolutionStrategy(
			ResolutionStrategy(inputSize, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER)
		).build()
}
