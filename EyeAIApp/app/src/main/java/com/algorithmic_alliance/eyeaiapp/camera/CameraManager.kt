package com.algorithmic_alliance.eyeaiapp.camera

import android.content.Context
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.TorchState
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import com.algorithmic_alliance.eyeaiapp.EyeAIApp.Companion.APP_LOG_TAG
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors

/**
 * Helper class that manages opening the camera using Android CameraX API
 * and hooks the [CameraFrameAnalyzer] up to the camera feed
 */
class CameraManager {
	private var cameraProviderListenableFuture: ListenableFuture<ProcessCameraProvider>? = null
	private var camera: Camera? = null
	var cameraPreview: Preview? = null
	private var depthAnalysisView: ImageAnalysis? = null

	fun init(
		context: Context,
		preferredImageSize: Size,
		cameraPreviewView: PreviewView?,
		cameraFrameAnalyzer: CameraFrameAnalyzer
	) {
		val lifecycleOwner = context as LifecycleOwner

		cameraProviderListenableFuture = ProcessCameraProvider.getInstance(context)

		cameraProviderListenableFuture?.addListener(
			{
				try {
					val cameraProvider: ProcessCameraProvider =
						cameraProviderListenableFuture!!.get()

					if (!lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
						camera = null
					}

					if (cameraPreview != null) cameraProvider.unbind(cameraPreview)

					if (depthAnalysisView != null) cameraProvider.unbind(depthAnalysisView)

					cameraPreview =
						Preview.Builder().setTargetFrameRate(Range<Int>(60, 120)).build()
					cameraPreview!!.surfaceProvider = cameraPreviewView!!.surfaceProvider

					depthAnalysisView =
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
					depthAnalysisView!!.setAnalyzer(
						Executors.newCachedThreadPool(),
						cameraFrameAnalyzer
					)

					camera = cameraProvider.bindToLifecycle(
						lifecycleOwner,
						mostWideCameraSelector(cameraProvider),
						depthAnalysisView,
						cameraPreview
					)

				} catch (e: ExecutionException) {
					Log.e(APP_LOG_TAG, e.message!!)
				} catch (e: InterruptedException) {
					Log.e(APP_LOG_TAG, e.message!!)
				}
			},
			ContextCompat.getMainExecutor(context)
		)
	}

	private fun hasCameraFlashlight(): Boolean {
		return camera?.cameraInfo?.hasFlashUnit() == true
	}

	fun isCameraFlashlightOn(): Boolean {
		if (!hasCameraFlashlight())
			return false

		return camera!!.cameraInfo.torchState.value == TorchState.ON
	}

	// toggles the camera flashlight and returns whether the flashlight was turned on
	fun toggleCameraFlashlight(): Boolean {
		if (!hasCameraFlashlight())
			return false

		val newFlashlightState = !isCameraFlashlightOn()
		camera!!.cameraControl.enableTorch(newFlashlightState)
		return newFlashlightState
	}
}