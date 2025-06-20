package com.algorithmic_alliance.eyeaiapp.camera

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

/**
 * Helper class that analyses the camera feed images in realtime
 */
@SuppressLint("SetTextI18n")
class CameraFrameAnalyzer(
	private var eyeAIApp: EyeAIApp,
	private var depthView: ImageView,
	private var performanceText: TextView,
) : ImageAnalysis.Analyzer {

	private var depthProcessingExecutor = Executors.newSingleThreadExecutor()
	private var objectDetectionProcessingExecutor = Executors.newSingleThreadExecutor()
	private var latestCameraFrame = AtomicReference<Bitmap?>(null)

	init {
		// DepthAnalyzer
		CoroutineScope(depthProcessingExecutor.asCoroutineDispatcher()).launch {
			while (isActive) {
				val depthModel = eyeAIApp.depthModel

				val frame = latestCameraFrame.getAndSet(null) // TODO: Verhindern dass Bilder "gestohlen" werden

				if (frame != null && depthModel != null) {
					NativeLib.newDepthFrame()

					val predictionOutput = depthModel.predictDepth(frame)

					val inputWidth = frame.width
					val inputHeight = frame.height

					withContext(Dispatchers.Main) {
						val colorMappedImage = NativeLib.depthColorMap(
							predictionOutput,
							depthModel.getInputSize()
						)
						depthView.setImageBitmap(colorMappedImage)

						if (eyeAIApp.settings.showProfilingInfo) {
							val formattedInputResolution = "${inputWidth}x${inputHeight}"
							val modelName = depthModel.getName()
							val modelInputSize = depthModel.getInputSize()
							val formattedModelInputSize =
								"${modelInputSize.width}x${modelInputSize.height}"
							performanceText.text =
								"Model: $modelName\nCamera resolution: $formattedInputResolution --> Model input: $formattedModelInputSize\n\n${NativeLib.formatDepthFrame()}\n${NativeLib.formatCameraFrame()}"
						} else {
							performanceText.text = ""
						}
					}
				}
			}
		}

		// Objekterkennung
		CoroutineScope(objectDetectionProcessingExecutor.asCoroutineDispatcher()).launch {
			while (isActive) {
				// val depthModel = eyeAIApp.depthModel

				val frame = latestCameraFrame.getAndSet(null)

				if (frame != null) {
					Log.e("object-recognition", "passed")
					// Frame analysieren
				}
			}
		}
	}

	@OptIn(ExperimentalGetImage::class)
	override fun analyze(image: ImageProxy) {
		if (image.image != null) {
			NativeLib.newCameraFrame()

			val inputBitmap =
				NativeLib.imageToBitmap(image.image!!, image.imageInfo.rotationDegrees.toFloat())

			latestCameraFrame.set(inputBitmap)
		}
		image.close()
	}
}
