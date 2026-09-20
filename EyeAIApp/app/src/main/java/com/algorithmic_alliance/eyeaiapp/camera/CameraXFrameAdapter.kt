package com.algorithmic_alliance.eyeaiapp.camera

import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib

/**
 * CameraX-only acquisition adapter. It converts and closes ImageProxy here;
 * all actual analysis happens in the source-neutral [FrameAnalyzer].
 */
class CameraXFrameAdapter(
	private val frameAnalyzer: FrameAnalyzer,
	private val cameraCalibration: CameraCalibration? = null,
) : ImageAnalysis.Analyzer {
	@OptIn(ExperimentalGetImage::class)
	override fun analyze(image: ImageProxy) {
		try {
			val mediaImage = image.image ?: return
			val rotationDegrees = image.imageInfo.rotationDegrees
			val bitmap = NativeLib.imageToBitmap(mediaImage, rotationDegrees.toFloat())
			frameAnalyzer.recordSourceFrame(image.imageInfo.timestamp)
			frameAnalyzer.submitFrame(
				AnalysisFrame(
					bitmap = bitmap,
					width = bitmap.width,
					height = bitmap.height,
					rotationDegrees = rotationDegrees,
					timestampNanos = image.imageInfo.timestamp,
					cameraIntrinsics = cameraCalibration?.forFrame(
						widthPx = bitmap.width,
						heightPx = bitmap.height,
						rotationDegrees = rotationDegrees,
					),
				)
			)
		} catch (error: Throwable) {
			Log.e(EyeAIApp.APP_LOG_TAG, "CameraX frame conversion failed", error)
		} finally {
			// The bitmap is detached from the ImageProxy before this point.
			image.close()
		}
	}
}
