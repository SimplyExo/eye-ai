package com.algorithmic_alliance.eyeaiapp.camera

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOCR
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOD
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
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
	private var overlay_od: OverlayViewOD,
	private var overlay_ocr: OverlayViewOCR
) : ImageAnalysis.Analyzer {

	private var depthProcessingExecutor = Executors.newSingleThreadExecutor()
	private var objectDetectionProcessingExecutor = Executors.newSingleThreadExecutor()
	private var ocrProcessingExecutor = Executors.newSingleThreadExecutor()

	private val depthScope = CoroutineScope(depthProcessingExecutor.asCoroutineDispatcher())
	private val objectScope = CoroutineScope(objectDetectionProcessingExecutor.asCoroutineDispatcher())
	private val ocrScope = CoroutineScope(ocrProcessingExecutor.asCoroutineDispatcher())

	private var latestCameraFrame = AtomicReference<Bitmap?>(null)

	private lateinit var colorMappedImage: Bitmap

	fun start() {
		// DepthAnalyzer
		depthScope.launch {
			while (isActive) {
				val depthModel = eyeAIApp.depthModel
				val frame = latestCameraFrame.get()

				if (frame != null && depthModel != null) {
					NativeLib.newDepthFrame()

					val predictionOutput = depthModel.predictDepth(frame)
					eyeAIApp.aiData.depthEstimationData.set(predictionOutput)

					val inputWidth = frame.width
					val inputHeight = frame.height

					withContext(Dispatchers.Main) {
						colorMappedImage = NativeLib.depthColorMap(
							predictionOutput,
							depthModel.inputDim
						)

						depthView.setImageBitmap(colorMappedImage)

						if (eyeAIApp.settings.showProfilingInfo) {
							val formattedInputResolution = "${inputWidth}x${inputHeight}"
							val modelName = depthModel.name
							val modelInputSize = depthModel.inputDim
							val formattedModelInputSize =
								"${modelInputSize.width}x${modelInputSize.height}"
							performanceText.text =
								"Model: $modelName\nCamera resolution: $formattedInputResolution --> Model input: $formattedModelInputSize\n\n${NativeLib.formatDepthFrame()}\n${NativeLib.formatCameraFrame()}\n${NativeLib.formatObjectFrame()}"
						} else {
							performanceText.text = ""
						}
					}
				}
			}
		}

		// Objekterkennung
		objectScope.launch {
			while (isActive) {
				if (eyeAIApp.settings.enableObjectDetection) {
					val frame = latestCameraFrame.get()

					if (frame != null) {
						NativeLib.newObjectFrame()

						// Frame analysieren
						val boxes = eyeAIApp.yoloModel?.runInference(frame)
						eyeAIApp.aiData.objectDetectionBoxes.set(boxes)

						// Anzeigen der Boxes
						if (boxes != null) {
							withContext(Dispatchers.Main) {
								overlay_od.setResults(boxes)
								overlay_od.setCameraResolution(
									Size(frame.width, frame.height)
								)
							}
						} else {
							withContext(Dispatchers.Main) {
								overlay_od.reset()
							}
						}
					}
				} else {
					overlay_od.reset()
				}
			}
		}

		// OCR Texterkennung
		ocrScope.launch {
			while (isActive) {
				if (eyeAIApp.settings.enableOCR) {
					val frame = latestCameraFrame.get()
					if (frame != null) {
						val textBoxes = eyeAIApp.ocrModel.analyzeFrame(frame).toTypedArray()
						eyeAIApp.aiData.ocrBoxes.set(textBoxes)
						overlay_ocr.setCameraResolution(
							Size(frame.width, frame.height)
						)
						overlay_ocr.setResults(textBoxes)
					}
				} else {
					overlay_ocr.reset()
				}
			}
		}
	}

	fun shutdown(timeoutMillis: Long = 1000) {
		depthScope.cancel()
		objectScope.cancel()
		ocrScope.cancel()

		try {
			// Warten auf Beendigung (maximal timeoutMillis)
			if (!depthProcessingExecutor.awaitTermination(timeoutMillis, java.util.concurrent.TimeUnit.MILLISECONDS)) {
				depthProcessingExecutor.shutdownNow()
			}
			if (!objectDetectionProcessingExecutor.awaitTermination(timeoutMillis, java.util.concurrent.TimeUnit.MILLISECONDS)) {
				objectDetectionProcessingExecutor.shutdownNow()
			}
			if (!ocrProcessingExecutor.awaitTermination(timeoutMillis, java.util.concurrent.TimeUnit.MILLISECONDS)) {
				ocrProcessingExecutor.shutdownNow()
			}
		} catch (e: InterruptedException) {
			// Im Fehlerfall sofort hart abbrechen
			depthProcessingExecutor.shutdownNow()
			objectDetectionProcessingExecutor.shutdownNow()
			ocrProcessingExecutor.shutdownNow()
			Thread.currentThread().interrupt() // Thread-Flag setzen
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
