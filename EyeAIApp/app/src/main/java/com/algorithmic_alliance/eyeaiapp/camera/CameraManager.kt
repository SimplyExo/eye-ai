package com.algorithmic_alliance.eyeaiapp.camera

import android.content.Context
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraProvider
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.TorchState
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.algorithmic_alliance.eyeaiapp.EyeAIApp.Companion.APP_LOG_TAG
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Helper class that manages opening the camera using Android CameraX API
 * and hooks the [CameraFrameAnalyzer] up to the camera feed
 */
class CameraManager {
	private var camera: Camera? = null
	var cameraFrameAnalyzer: CameraFrameAnalyzer? = null
	private var cameraFrameAnalyzerExecutor: ExecutorService = Executors.newSingleThreadExecutor()

	fun init(
		context: Context,
		lifecycleOwner: LifecycleOwner,
		preferredImageSize: Size,
		cameraPreviewView: PreviewView?,
	) {
		val cameraProviderListenableFuture = ProcessCameraProvider.getInstance(context)

		cameraProviderListenableFuture.addListener(
			{
				try {
					val cameraProvider: ProcessCameraProvider =
						cameraProviderListenableFuture.get()

					val cameraPreview =
						Preview.Builder().setTargetFrameRate(Range(60, 120)).build()

					val analysisView =
						ImageAnalysis.Builder()
							.setImageQueueDepth(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
							.setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
							.setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
							.setResolutionSelector(
								performanceResolutionSelector(
									preferredImageSize
								)
							)
							.build()
					analysisView.setAnalyzer(
						cameraFrameAnalyzerExecutor,
						cameraFrameAnalyzer!!
					)

					cameraProvider.unbindAll()
					cameraProvider.bindToLifecycle(
						lifecycleOwner,
						mostWideCameraSelector(cameraProvider),
						analysisView,
						cameraPreview
					)

					cameraPreview.surfaceProvider = cameraPreviewView!!.surfaceProvider
				} catch (e: ExecutionException) {
					Log.e(APP_LOG_TAG, e.message!!)
				} catch (e: InterruptedException) {
					Log.e(APP_LOG_TAG, e.message!!)
				}
			},
			ContextCompat.getMainExecutor(context)
		)
	}

	fun shutdown() {
		cameraFrameAnalyzer?.shutdown()
		cameraFrameAnalyzerExecutor.apply {
			shutdown()
			awaitTermination(1000, TimeUnit.MILLISECONDS)
		}
	}

	fun pauseAnalyzer() {
		if (cameraFrameAnalyzer?.started == true) {
			cameraFrameAnalyzer?.shutdown()
		}
	}

	fun resumeAnalyzer() {
		if (cameraFrameAnalyzer?.started == false) {
			cameraFrameAnalyzer?.start()
		}
	}
}