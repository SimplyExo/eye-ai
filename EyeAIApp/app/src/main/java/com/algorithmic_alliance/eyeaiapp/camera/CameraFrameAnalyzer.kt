package com.algorithmic_alliance.eyeaiapp.camera

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
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
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import androidx.core.view.isVisible
import java.util.concurrent.ExecutorService

/**
 * Helper class that analyses the camera feed images in realtime
 */
@SuppressLint("SetTextI18n")
class CameraFrameAnalyzer(
	private var eyeAIApp: EyeAIApp,
	private var depthView: ImageView,
	private var performanceText: TextView,
	private var overlayOD: OverlayViewOD,
	private var overlayOCR: OverlayViewOCR,
	private var debugInputBitmapPreview: ImageView,
	private var mediaImageView: ImageView
) : ImageAnalysis.Analyzer {

	private var depthProcessingExecutor = Executors.newSingleThreadExecutor()
	private var objectDetectionProcessingExecutor: ExecutorService? = null
	private var ocrProcessingExecutor: ExecutorService? = null

	private val depthScope: CoroutineScope =
		CoroutineScope(depthProcessingExecutor.asCoroutineDispatcher())
	private val objectScope: CoroutineScope? = if (eyeAIApp.settings.enableObjectDetection) {
		objectDetectionProcessingExecutor = Executors.newSingleThreadExecutor()
		CoroutineScope(objectDetectionProcessingExecutor!!.asCoroutineDispatcher())
	} else {
		null
	}
	private val ocrScope = if (eyeAIApp.settings.enableOCR) {
		ocrProcessingExecutor = Executors.newSingleThreadExecutor()
		CoroutineScope(ocrProcessingExecutor!!.asCoroutineDispatcher())
	} else {
		null
	}

	private var latestCameraFrame = AtomicReference<Bitmap?>(null)

	private lateinit var colorMappedImage: Bitmap

	var started = false
		private set

	fun start() {
		started = true

		// DepthAnalyzer
		depthScope.launch {
			while (isActive) {
				val metricDepthModel = eyeAIApp.metricDepthModel
				val frame = getFrame()

				if (frame != null && metricDepthModel != null) {
					NativeLib.newDepthFrame()

					val predictionOutput = metricDepthModel.predictDepth(frame)
					eyeAIApp.aiData.depthEstimationData.set(predictionOutput)

					val inputWidth = frame.width
					val inputHeight = frame.height

					colorMappedImage = NativeLib.metricDepthColormap(
						predictionOutput,
						metricDepthModel.inputDim
					)

					withContext(Dispatchers.Main) {
						depthView.setImageBitmap(colorMappedImage)

						if (debugInputBitmapPreview.isVisible)
							debugInputBitmapPreview.setImageBitmap(frame)

						if (eyeAIApp.settings.showProfilingInfo) {
							val formattedInputResolution = "${inputWidth}x${inputHeight}"
							val formattedDepthModelInputSize =
								"${metricDepthModel.inputDim.width}x${metricDepthModel.inputDim.height}"
							performanceText.text =
								"Metric Depth model: ${metricDepthModel.name}\nCamera resolution: $formattedInputResolution --> Depth model input: $formattedDepthModelInputSize\n\n${NativeLib.formatDepthFrame()}\n${NativeLib.formatCameraFrame()}\n${NativeLib.formatObjectFrame()}"
						} else {
							performanceText.text = ""
						}
					}
				}
			}
		}

		// Objekterkennung
		objectScope?.launch {
			while (isActive) {
				val frame = getFrame()

				if (frame != null) {
					NativeLib.newObjectFrame()

					// Frame analysieren
					val boxes = eyeAIApp.yoloModel.runInference(frame)
					eyeAIApp.aiData.objectDetectionBoxes.set(boxes)

					// Anzeigen der Boxes
					withContext(Dispatchers.Main) {
						if (boxes != null) {
							overlayOD.setResults(boxes)
							overlayOD.setCameraResolution(Size(frame.width, frame.height))
						} else {
							overlayOD.reset()
						}
					}
				}
			}
		}

		// OCR Texterkennung
		ocrScope?.launch {
			while (isActive) {
				val frame = getFrame()

				if (frame != null) {
					val textBoxes = eyeAIApp.ocrModel.analyzeFrame(frame).toTypedArray()
					eyeAIApp.aiData.ocrBoxes.set(textBoxes)

					withContext(Dispatchers.Main) {
						overlayOCR.setCameraResolution(
							Size(frame.width, frame.height)
						)
						overlayOCR.setResults(textBoxes)
					}
				}
			}
		}
	}

	fun getFrame(): Bitmap? {
		val frame: Bitmap? = if (eyeAIApp.settings.inputSource == "camera") {
			latestCameraFrame.get()
		} else {
			if (mediaImageView.drawable == null) {
				null
			} else {
				(mediaImageView.drawable as BitmapDrawable).bitmap
			}
		}

		return frame
	}

	fun shutdown() {
		depthScope.cancel()
		objectScope?.cancel()
		ocrScope?.cancel()

		depthProcessingExecutor.shutdownNow()
		objectDetectionProcessingExecutor?.shutdownNow()
		ocrProcessingExecutor?.shutdownNow()

		started = false
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
