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

/**
 * @return Selector that selects the most wide angle sens back camera
 */
@OptIn(ExperimentalCamera2Interop::class)
fun mostWideCameraSelector(cameraProvider: ProcessCameraProvider): CameraSelector {
	var widestCamera: CameraInfo? = null
	var smallestFocalLength: Float? = null

	for (cameraInfo in cameraProvider.availableCameraInfos) {
		if (cameraInfo.lensFacing != CameraSelector.LENS_FACING_BACK) {
			continue
		}

		val camera2CameraInfo = Camera2CameraInfo.from(cameraInfo)
		val focalLengths =
			camera2CameraInfo.getCameraCharacteristic(
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

	return widestCamera?.cameraSelector ?: CameraSelector.DEFAULT_BACK_CAMERA
}

/**
 * @return Selector that chooses the smallest possible resolution that still fits the [inputSize]
 */
fun performanceResolutionSelector(inputSize: Size): ResolutionSelector {
	return ResolutionSelector.Builder()
		.setAllowedResolutionMode(ResolutionSelector.PREFER_CAPTURE_RATE_OVER_HIGHER_RESOLUTION)
		.setResolutionStrategy(
			ResolutionStrategy(inputSize, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER)
		)
		.build()
}
