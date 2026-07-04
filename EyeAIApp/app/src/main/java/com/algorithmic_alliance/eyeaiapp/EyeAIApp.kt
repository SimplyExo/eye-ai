package com.algorithmic_alliance.eyeaiapp

import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudio
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModel
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModelInfo
import com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio.GoogleAIStudioLLM
import com.algorithmic_alliance.eyeaiapp.llm.LLM
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModel
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModelInfo
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModel
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModelInfo
import com.algorithmic_alliance.eyeaiapp.ocr.GoogleOCR
import com.algorithmic_alliance.eyeaiapp.speech_recognition.VoskModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.Executors

/**
 * App class that holds everything that should persist when switching to another app, for example
 * the camera handle and the loaded depth model
 */
class EyeAIApp : Application() {
	lateinit var settings: Settings
		private set

	private var loadDepthModelExecutor = Executors.newSingleThreadExecutor { r ->
		Thread(r, "Depth loader")
	}
	private var loadYoloModelExecutor = Executors.newSingleThreadExecutor { r ->
		Thread(r, "Yolo loader")
	}
	private var loadNlpModelExecutor = Executors.newSingleThreadExecutor { r ->
		Thread(r, "Nlp loader")
	}

	var metricDepthModel: MetricDepthModel? = null
		private set

	/* will not load the model or listen if enableSpeechRecognition is disabled in settings, needs to be started manually inside MainActivity */
	lateinit var voskModel: VoskModel
		private set

	/* can be [null] if googleAiStudioApiKey is not set in settings */
	var llm: LLM? = null
		private set

	/* will not be fully created if enableObjectDetection is disabled in settings */
	var yoloModel: YoloModel = YoloModel(YoloModelInfo("model.tflite", "coco.names", 640))
		private set

	var nlpModel: NLPModel = NLPModel(NLPModelInfo("nlp_model_float32.tflite"))
		private set

	/* will not be fully initialized when enableOCR is disabled in settings */
	var ocrModel = GoogleOCR()
		private set

	var aiData = AIModelData

	var npuQnnDelegateDirectory: String? = null

	companion object {
		const val APP_LOG_TAG = "Eye AI"

		const val DEFAULT_DEPTH_MODEL_NAME = "MiDaS V2.1"

		val DEPTH_MODELS = arrayOf(
			MetricDepthModelInfo(
				DEFAULT_DEPTH_MODEL_NAME, "midas_v2_1_256x256.tflite"
			), MetricDepthModelInfo(
				"MiDaS V2.1 (quantized)", "midas_v2_1_256x256_quantized.tflite"
			)
		)

		val PREFERRED_CAMERA_RESOLUTION = Size(640, 640)
	}

	override fun onCreate() {
		super.onCreate()

		uniffi.NativeLib.initAndroidLogging()

		val context = this

		settings = Settings.load(context)

		// does not load model or start listening
		voskModel = VoskModel(context, "model-de")

		settings.googleAiStudioApiKey?.let { apiKey ->
			if (!apiKey.isEmpty()) llm =
				GoogleAIStudioLLM(apiKey, settings.customGoogleGenAIStudioEndpoint)
		}

		npuQnnDelegateDirectory = applicationInfo.nativeLibraryDir

		CoroutineScope(loadDepthModelExecutor.asCoroutineDispatcher()).launch {
			switchDepthModel(settings.depthModel)
		}

		CoroutineScope(loadYoloModelExecutor.asCoroutineDispatcher()).launch {
			// Yolo Model erstellen
			if (settings.enableObjectDetection) {
				yoloModel.create(baseContext, npuQnnDelegateDirectory!!, settings.enableNpu)
			}
		}

		CoroutineScope(loadNlpModelExecutor.asCoroutineDispatcher()).launch {
			// NLP erstellen
			nlpModel.create(baseContext)
		}

		// Google ML Kit initialisieren
		if (settings.enableOCR) ocrModel.create()
	}

	fun updateSettings() {
		val context = this as Context
		val oldSettings = settings.clone()
		settings = Settings.load(context)

		if (oldSettings.depthAudioPlayback != settings.depthAudioPlayback) {
			uniffi.NativeLib.setDepthAudioPaused(!settings.depthAudioPlayback)
		}

		if (oldSettings.objectAudioPlayback != settings.objectAudioPlayback) {
			uniffi.NativeLib.setObjectAudioPaused(!settings.objectAudioPlayback)
		}

		if (oldSettings.depthAudioFrequency != settings.depthAudioFrequency) {
			uniffi.NativeLib.setAudioSettings(
				settings.depthAudioFrequency.toFloat(), settings.depthAudioClickIncidence
			)
		}

		if (oldSettings.depthAudioClickIncidence != settings.depthAudioClickIncidence) {
			uniffi.NativeLib.setAudioSettings(
				settings.depthAudioFrequency.toFloat(), settings.depthAudioClickIncidence
			)
		}

		if (oldSettings.objectAudioPlaybackLanguage != settings.objectAudioPlaybackLanguage) {
			// TODO: move this somewhere else?
			SpatialAudio.stop()
			CoroutineScope(Dispatchers.IO).launch {
				SpatialAudio.setup(this@EyeAIApp)
				SpatialAudio.start()
			}
		}

		val enableNpuChanged = oldSettings.enableNpu != settings.enableNpu

		CoroutineScope(loadDepthModelExecutor.asCoroutineDispatcher()).launch {
			if (oldSettings.depthModel != settings.depthModel || enableNpuChanged) {
				switchDepthModel(settings.depthModel)
			}
		}

		if (oldSettings.enableSpeechRecognition != settings.enableSpeechRecognition) {
			if (!settings.enableSpeechRecognition) {
				voskModel.closeService()
			}
		}

		if (oldSettings.googleAiStudioApiKey != settings.googleAiStudioApiKey || oldSettings.customGoogleGenAIStudioEndpoint != settings.customGoogleGenAIStudioEndpoint) {
			val apiKey = settings.googleAiStudioApiKey
			val customEndpoint = settings.customGoogleGenAIStudioEndpoint
			llm = if (!apiKey.isNullOrEmpty()) {
				GoogleAIStudioLLM(apiKey, customEndpoint)
			} else {
				null
			}
		}


		CoroutineScope(loadDepthModelExecutor.asCoroutineDispatcher()).launch {
			if (oldSettings.enableObjectDetection != settings.enableObjectDetection || enableNpuChanged) {
				if (settings.enableObjectDetection) {
					yoloModel.create(baseContext, npuQnnDelegateDirectory!!, settings.enableNpu)
				}
			}
		}

		if (oldSettings.enableOCR != settings.enableOCR) {
			if (settings.enableOCR) {
				ocrModel.create()
			}
		}
	}

	private fun switchDepthModel(modelName: String) {
		if (metricDepthModel?.name == modelName && metricDepthModel?.enableNpu == settings.enableNpu) return

		metricDepthModel?.close()
		metricDepthModel = null

		metricDepthModel = findDepthModelInfo(modelName).createDepthModel(
			this, npuQnnDelegateDirectory!!, settings.enableNpu
		)
	}

	private fun findDepthModelInfo(modelName: String): MetricDepthModelInfo {
		return DEPTH_MODELS.find { it.name == modelName }
			?: (DEPTH_MODELS.find { it.name == DEFAULT_DEPTH_MODEL_NAME } ?: DEPTH_MODELS[0])
	}
}
