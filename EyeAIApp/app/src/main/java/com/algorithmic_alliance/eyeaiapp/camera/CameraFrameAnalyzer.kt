package com.algorithmic_alliance.eyeaiapp.camera

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
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
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import androidx.core.view.isVisible
import kotlinx.coroutines.delay
import java.util.concurrent.ExecutorService
import kotlin.time.Duration.Companion.seconds
import kotlin.time.measureTime

/**
 * Helper class that analyses the camera feed images in realtime
 */
@SuppressLint("SetTextI18n")
class CameraFrameAnalyzer(
	private var eyeAIApp: EyeAIApp,
	private var depthView: ImageView,
	private var performanceText: TextView,
	private var overlayOD: OverlayViewOD,
	private var debugInputBitmapPreview: ImageView,
	private var mediaImageView: ImageView
) : ImageAnalysis.Analyzer {

	private var depthProcessingExecutor = Executors.newSingleThreadExecutor()
	private var objectDetectionProcessingExecutor: ExecutorService? = null

	private val depthScope: CoroutineScope =
		CoroutineScope(depthProcessingExecutor.asCoroutineDispatcher())

	private val objectScope: CoroutineScope? = if (eyeAIApp.settings.enableObjectDetection) {
		objectDetectionProcessingExecutor = Executors.newSingleThreadExecutor()
		CoroutineScope(objectDetectionProcessingExecutor!!.asCoroutineDispatcher())
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
					val inferenceDuration = measureTime {
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
									"Metric Depth model: ${metricDepthModel.name}\nCamera resolution: $formattedInputResolution -> Depth model input: $formattedDepthModelInputSize\n\n${NativeLib.formatDepthFrame()}\n${NativeLib.formatCameraFrame()}\n${NativeLib.formatObjectFrame()}"
							} else {
								performanceText.text = ""
							}
						}
					}

					val maxFrameRate = 30 //eyeAIApp.settings.maxDepthFrameRate
					val minInferenceDuration =(maxFrameRate?.let { 1.0 / it })?.seconds
					if (minInferenceDuration != null && inferenceDuration < minInferenceDuration) {
						delay(minInferenceDuration - inferenceDuration)
					}
				}
			}
		}

		// object-detection
		objectScope?.launch {
			while (isActive) {
				val frame = getFrame()
				if (frame != null) {
					val inferenceDuration = measureTime {
						NativeLib.newObjectFrame()
						// analyzing the frame
						val boxes = eyeAIApp.yoloModel.runInference(frame)
						eyeAIApp.aiData.objectDetectionBoxes.set(boxes)

						// showing boxes
						withContext(Dispatchers.Main) {
							if (boxes != null) {
								overlayOD.setResults(boxes)
								overlayOD.setCameraResolution(Size(frame.width, frame.height))
							} else {
								overlayOD.reset()
							}
						}
					}

					val maxFrameRate = 30 //eyeAIApp.settings.maxObjectDetectionFrameRate TODO: Fix this as well
					val minInferenceDuration = (maxFrameRate?.let { 1.0 / it })?.seconds
					if (minInferenceDuration != null && inferenceDuration < minInferenceDuration) {
						delay(minInferenceDuration - inferenceDuration)
					}
				}
			}
		}

		// OCR removed, on demand only now.
	}

	suspend fun runOcrAnalysis(): Boolean = withContext(Dispatchers.IO) {
		val frame = getFrame() ?: return@withContext false

		return@withContext try {
			Log.d(EyeAIApp.APP_LOG_TAG, "Running on-demand OCR analysis")
			val textBoxes = eyeAIApp.ocrModel.analyzeFrame(frame).toTypedArray()
			eyeAIApp.aiData.ocrBoxes.set(textBoxes)

			// OCR-Overlay removed

			Log.d(EyeAIApp.APP_LOG_TAG, "OCR analysis completed successfully, found ${textBoxes.size} text boxes")
			true
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Error during on-demand OCR analysis", e)
			false
		}
	}

	fun getFrame(): Bitmap? {
		return if (eyeAIApp.settings.inputSource == "camera") {
			latestCameraFrame.get()
		} else {
			(mediaImageView.drawable as? BitmapDrawable)?.bitmap
		}
	}

	fun shutdown() {
		depthScope.cancel()
		objectScope?.cancel()
		depthProcessingExecutor.shutdownNow()
		objectDetectionProcessingExecutor?.shutdownNow()
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
