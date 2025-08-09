package com.algorithmic_alliance.eyeaiapp

import android.app.Application
import android.content.Context
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.depth.DepthModel
import com.algorithmic_alliance.eyeaiapp.depth.DepthModelInfo
import com.algorithmic_alliance.eyeaiapp.llm.GoogleAIStudioLLM
import com.algorithmic_alliance.eyeaiapp.llm.LLM
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModel
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModelInfo
import com.algorithmic_alliance.eyeaiapp.ocr.GoogleOCR
import com.algorithmic_alliance.eyeaiapp.speech_recognition.VoskModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

/**
 * App class that holds everything that should persist when switching to another app, for example
 * the camera handle and the loaded depth model
 */
class EyeAIApp : Application() {
	lateinit var settings: Settings
		private set

	private var loadAIModelExecutor = Executors.newSingleThreadExecutor()

	var depthModel: DepthModel? = null
		private set

	/* will not load the model or listen if enableSpeechRecognition is disabled in settings, needs to be started manually inside MainActivity */
	lateinit var voskModel: VoskModel
		private set

	/* can be [null] if googleAiStudioApiKey is not set in settings */
	var llm: LLM? = null
		private set

	/* will not be fully created if enableObjectDetection is disabled in settings */
	var yoloModel: YoloModel = YoloModel(YoloModelInfo("model.tflite", 640))
		private set

	/* will not be fully initialized when enableOCR is disabled in settings */
	var ocrModel = GoogleOCR()
		private set

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

		val PREFERRED_CAMERA_RESOLUTION = Size(640, 640)
	}

	override fun onCreate() {
		super.onCreate()

		val context = this

		settings = Settings.load(context)

		// does not load model or start listening
		voskModel = VoskModel(context, "model-de")

		CoroutineScope(loadAIModelExecutor.asCoroutineDispatcher()).launch {
			switchDepthModel(settings.depthModel)

			settings.googleAiStudioApiKey?.let { apiKey ->
				if (!apiKey.isEmpty())
					llm = GoogleAIStudioLLM(apiKey)
			}

			// Yolo Model erstellen
			if (settings.enableObjectDetection) {
				yoloModel.create(baseContext)
			}

			// Google ML Kit initialisieren
			if (settings.enableOCR)
				ocrModel.create()
		}
	}

	fun updateSettings() {
		val context = this as Context
		val oldSettings = settings.clone()
		settings = Settings.load(context)

		CoroutineScope(loadAIModelExecutor.asCoroutineDispatcher()).launch {
			if (oldSettings.depthModel != settings.depthModel) {
				switchDepthModel(settings.depthModel)
			}

			if (oldSettings.enableSpeechRecognition != settings.enableSpeechRecognition) {
				if (!settings.enableSpeechRecognition) {
					voskModel.closeService()
				}
			}

			if (oldSettings.googleAiStudioApiKey != settings.googleAiStudioApiKey) {
				val apiKey = settings.googleAiStudioApiKey
				llm = if (apiKey != null && !apiKey.isEmpty()) {
					GoogleAIStudioLLM(apiKey)
				} else {
					null
				}
			}

			if (oldSettings.enableObjectDetection != settings.enableObjectDetection) {
				if (settings.enableObjectDetection) {
					yoloModel.create(baseContext)
				}
			}

			if (oldSettings.enableOCR != settings.enableOCR) {
				if (settings.enableOCR) {
					ocrModel.create()
				}
			}
		}
	}

	private fun switchDepthModel(modelName: String) {
		if (depthModel?.name == modelName) return

		depthModel?.close()
		depthModel = null

		depthModel = findDepthModelInfo(modelName)
			.createDepthModel(this)
	}

	private fun findDepthModelInfo(modelName: String): DepthModelInfo {
		return DEPTH_MODELS.find { it.name == modelName }
			?: (DEPTH_MODELS.find { it.name == DEFAULT_DEPTH_MODEL_NAME } ?: DEPTH_MODELS[0])
	}
}
