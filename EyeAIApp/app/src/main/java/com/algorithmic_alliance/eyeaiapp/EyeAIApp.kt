package com.algorithmic_alliance.eyeaiapp

import android.app.Application
import android.content.Context
import android.util.Log
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.depth.DepthModel
import com.algorithmic_alliance.eyeaiapp.depth.DepthModelInfo
import com.algorithmic_alliance.eyeaiapp.llm.GoogleAIStudioLLM
import com.algorithmic_alliance.eyeaiapp.llm.LLM
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModel
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModelInfo
import com.algorithmic_alliance.eyeaiapp.ocr.GoogleOCR
import com.algorithmic_alliance.eyeaiapp.speech_recognition.VoskModel
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.internal.TextRecognizerOptionsUtils
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * App class that holds everything that should persist when switching to another app, for example
 * the camera handle and the loaded depth model
 */
class EyeAIApp : Application() {
	lateinit var settings: Settings
		private set
	var depthModel: DepthModel? = null
		private set
	var onDepthModelLoadedCallback: () -> Unit = {}

	/* can be [null] if enableSpeechRecognition is disabled in settings */
	var voskModel: VoskModel? = null
		private set
	var llm: LLM? = null
		private set

	var yoloModel: YoloModel? = null
		private set

	var ocrModel = GoogleOCR()

	var aiData = AIModelData

	companion object {
		const val APP_LOG_TAG = "Eye AI"

		const val DEFAULT_DEPTH_MODEL_NAME = "MiDaS V2.1"

		val DEPTH_MODELS =
			arrayOf(
				DepthModelInfo(
					DEFAULT_DEPTH_MODEL_NAME,
					"midas_v2_1_256x256.tflite"
				),
				DepthModelInfo(
					"MiDaS V2.1 (quantized)",
					"midas_v2_1_256x256_quantized.tflite"
				)
			)
	}

	override fun onCreate() {
		super.onCreate()

		settings = Settings(this)

		switchDepthModel(settings.depthModel)

		if (settings.enableSpeechRecognition)
			voskModel = VoskModel(this, "model-de")

		settings.googleAiStudioApiKey?.let {
			if (!it.isEmpty())
				llm = GoogleAIStudioLLM(it)
		}

		// Yolo Model erstellen
		yoloModel = YoloModel(YoloModelInfo("model.tflite", 640))
		yoloModel!!.create(baseContext)

		// Google ML Kit initialisieren
		ocrModel.create()
	}

	fun getPreferredCameraResolution(): Size? {
		val depthResolution = depthModel?.inputDim ?: return null
		val objectSize = yoloModel?.info?.size ?: return null

		return Size(
			maxOf(depthResolution.width, objectSize), maxOf(depthResolution.height, objectSize)
		)
	}

	fun updateSettings() {
		val newSettings = Settings(this)

		if (settings.depthModel != newSettings.depthModel) {
			switchDepthModel(newSettings.depthModel)
		}

		if (settings.enableSpeechRecognition != newSettings.enableSpeechRecognition) {
			val context = this as Context
			CoroutineScope(Dispatchers.IO).launch {
				if (newSettings.enableSpeechRecognition) {
					voskModel = VoskModel(context, "model-de")
				} else {
					voskModel?.closeService()
					voskModel = null
				}
			}
		}

		if (settings.googleAiStudioApiKey != newSettings.googleAiStudioApiKey) {
			val apiKey = newSettings.googleAiStudioApiKey
			llm = if (apiKey != null && !apiKey.isEmpty()) {
				GoogleAIStudioLLM(apiKey)
			} else {
				null
			}
		}

		settings = newSettings
	}

	private fun switchDepthModel(modelName: String) {
		if (depthModel?.name == modelName) return

		depthModel?.close()
		depthModel = null

		val context = this as Context
		CoroutineScope(Dispatchers.IO).launch {
			depthModel = findDepthModelInfo(modelName)
				.createDepthModel(context)

			if (depthModel != null) {
				withContext(Dispatchers.Main) {
					onDepthModelLoadedCallback()
				}
			} else {
				Log.e(
					APP_LOG_TAG,
					"Failed to init depth model $modelName"
				)
			}
		}
	}

	private fun findDepthModelInfo(modelName: String): DepthModelInfo {
		return DEPTH_MODELS.find { it.name == modelName }
			?: (DEPTH_MODELS.find { it.name == DEFAULT_DEPTH_MODEL_NAME } ?: DEPTH_MODELS[0])
	}
}
