package com.algorithmic_alliance.eyeaiapp.camera

import android.annotation.SuppressLint
import android.graphics.Bitmap
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
	private var overlay_od: OverlayViewOD,
	private var overlay_ocr: OverlayViewOCR,
	private var yoloInferenceDuration: TextView
) : ImageAnalysis.Analyzer {

	private var depthProcessingExecutor = Executors.newSingleThreadExecutor()
	private var objectDetectionProcessingExecutor = Executors.newSingleThreadExecutor()
	private var ocrProcessingExecutor = Executors.newSingleThreadExecutor()

	private var latestCameraFrame = AtomicReference<Bitmap?>(null)

	private lateinit var colorMappedImage: Bitmap

	init {
		// DepthAnalyzer
		CoroutineScope(depthProcessingExecutor.asCoroutineDispatcher()).launch {
			while (isActive) {
				val depthModel = eyeAIApp.depthModel
				val frame = latestCameraFrame.get()

				if (frame != null && depthModel != null) {
					NativeLib.newDepthFrame()

					val predictionOutput = depthModel.predictDepth(frame)

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
		CoroutineScope(objectDetectionProcessingExecutor.asCoroutineDispatcher()).launch {
			while (isActive) {
				val frame = latestCameraFrame.get()

				if (frame != null && eyeAIApp.settings.enableObjectDetection) {
					NativeLib.newObjectFrame()

					// Frame analysieren
					var inferenceTime = System.currentTimeMillis()
					val boxes = eyeAIApp.yoloModel?.runInference(frame);
					inferenceTime = System.currentTimeMillis() - inferenceTime

					// Verarbeiten und Anzeigen der Boxes
					if (boxes != null) {
						withContext(Dispatchers.Main) {
							yoloInferenceDuration.text = "$inferenceTime ms"
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
			}
		}

		// OCR Texterkennung
		CoroutineScope(ocrProcessingExecutor.asCoroutineDispatcher()).launch {
			while (isActive) {
				val frame = latestCameraFrame.get()

				if (frame != null) {
					val text_boxes = eyeAIApp.ocrModel.analyzeFrame(frame).toTypedArray()
					overlay_ocr.setCameraResolution(
						Size(frame.width, frame.height)
					)
					overlay_ocr.setResults(text_boxes)
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
