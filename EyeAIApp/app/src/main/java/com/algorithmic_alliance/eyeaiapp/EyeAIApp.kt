package com.algorithmic_alliance.eyeaiapp

import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.provider.Contacts
import android.util.Log
import android.util.Size
import androidx.activity.viewModels
import androidx.lifecycle.viewmodel.compose.viewModel
import com.algorithmic_alliance.eyeaiapp.UI.MainViewModel
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudio
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.connectivity.EyeAIVision
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModel
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModel
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModelInfo
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateMachine
import com.algorithmic_alliance.eyeaiapp.media.MediaPlayer
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModel
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModelInfo
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModel
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModelInfo
import com.algorithmic_alliance.eyeaiapp.ocr.GoogleOCR
import com.algorithmic_alliance.eyeaiapp.settingsparser.LocalSettingsParser
import com.algorithmic_alliance.eyeaiapp.speech_recognition.VoskModel
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as UI_LOG_TAG
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.launch
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.getValue

/**
 * App class that holds everything that should persist when switching to another app, for example
 * the camera handle and the loaded depth model
 */
class EyeAIApp : Application() {
	lateinit var settings: Settings
		private set

	val speechThreadExecutor = Executors.newSingleThreadExecutor()

	var lastDialogContext: String? = null

	private var loadAIModelExecutor = Executors.newSingleThreadExecutor()

	var metricDepthModel: MetricDepthModel? = null
		private set

	/* will not load the model or listen if enableSpeechRecognition is disabled in settings, needs to be started manually inside MainActivity */
	lateinit var voskModel: VoskModel
		private set

	/* will not be fully created if enableObjectDetection is disabled in settings */
	var yoloModel: YoloModel =
		YoloModel(YoloModelInfo("model.tflite", "coco.names", 640))
		private set

	var nlpModel: NLPModel =
		NLPModel(NLPModelInfo.findById(NLPModelInfo.DEFAULT_MODEL_ID))
		private set

	/** Loaded lazily and shared across short-lived StateMachine instances. */
	val confirmationModel: ConfirmationModel by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
		val started = System.nanoTime()
		Log.i(
			APP_LOG_TAG,
			"[DecisionTrace][ConfirmationModel][LOAD] outcome=STARTED " +
				"model=${ConfirmationModel.MODEL_ID} asset=${ConfirmationModel.ASSET_PATH} " +
				"execution=LOCAL apiCalled=false"
		)
		try {
			ConfirmationModel.fromAssets(this).also { model ->
				Log.i(
					APP_LOG_TAG,
					"[DecisionTrace][ConfirmationModel][LOAD] outcome=SUCCESS " +
						"model=${ConfirmationModel.MODEL_ID} featureCount=${model.featureCount} " +
						"threshold=${String.format(Locale.US, "%.4f", model.confidenceThreshold)} " +
						"duration=${(System.nanoTime() - started) / 1_000_000}ms " +
						"execution=LOCAL apiCalled=false"
				)
			}
		} catch (error: Throwable) {
			Log.e(
				APP_LOG_TAG,
				"[DecisionTrace][ConfirmationModel][LOAD] outcome=FAILED " +
					"model=${ConfirmationModel.MODEL_ID} " +
					"duration=${(System.nanoTime() - started) / 1_000_000}ms " +
					"execution=LOCAL apiCalled=false",
				error
			)
			throw error
		}
	}

	/**
	 * The two frozen Clean-v2 interpreters are loaded lazily once, from product
	 * assets only. SettingsHandler requests this property on a background dispatcher.
	 */
	private val localSettingsParserLazy = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
		val started = System.nanoTime()
		Log.i(
			APP_LOG_TAG,
			"[DecisionTrace][SettingsParser][LOAD] architecture=SPECIALIZED_WORD_OPERATION_CHAR_SPEAKER " +
				"execution=LOCAL apiCalled=false outcome=STARTED"
		)
		try {
			LocalSettingsParser.fromAssets(this).also {
				Log.i(
					APP_LOG_TAG,
					"[DecisionTrace][SettingsParser][LOAD] execution=LOCAL apiCalled=false " +
						"outcome=SUCCESS duration=${(System.nanoTime() - started) / 1_000_000}ms"
				)
			}
		} catch (error: Throwable) {
			Log.e(
				APP_LOG_TAG,
				"[DecisionTrace][SettingsParser][LOAD] execution=LOCAL apiCalled=false outcome=FAILED " +
					"duration=${(System.nanoTime() - started) / 1_000_000}ms",
				error
			)
			throw error
		}
	}

	val localSettingsParser: LocalSettingsParser
		get() = localSettingsParserLazy.value

	/* will not be fully initialized when enableOCR is disabled in settings */
	var ocrModel = GoogleOCR()
		private set

	lateinit var textToSpeechInstance: TextToSpeechInstance
	var lastLlmJsonResponse: String? = null
	var cameraManager = CameraManager()
	var mediaFrameAnalyzer: CameraFrameAnalyzer? = null
	var currentStateMachine: StateMachine? = null
	var mediaPlayer: MediaPlayer? = null
	var bitmapFlow: MutableSharedFlow<Bitmap>? = null
	lateinit var eyeAIVision: EyeAIVision
	var aiData = AIModelData

	var npuQnnDelegateDirectory: String? = null

	companion object {
		const val APP_LOG_TAG = "Eye AI"

		const val DEFAULT_DEPTH_MODEL_NAME = "MiDaS V2.1"

		val DEPTH_MODELS =
			arrayOf(
				MetricDepthModelInfo(
					DEFAULT_DEPTH_MODEL_NAME,
					"midas_v2_1_256x256.tflite"
				),
				MetricDepthModelInfo(
					"MiDaS V2.1 (quantized)",
					"midas_v2_1_256x256_quantized.tflite"
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

		npuQnnDelegateDirectory = applicationInfo.nativeLibraryDir

		CoroutineScope(loadAIModelExecutor.asCoroutineDispatcher()).launch {
			switchDepthModel(settings.depthModel)

			// Yolo Model erstellen
			if (settings.enableObjectDetection) {
				yoloModel.create(baseContext, npuQnnDelegateDirectory!!, settings.enableNpu)
			}

			// NLP erstellen
			switchNlpModel(settings.nlpModel)

			// Google ML Kit initialisieren
			if (settings.enableOCR)
				ocrModel.create()
		}
	}

	override fun onTerminate() {
		if (localSettingsParserLazy.isInitialized()) {
			localSettingsParserLazy.value.close()
		}
		super.onTerminate()
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
				settings.depthAudioFrequency.toFloat(),
				settings.depthAudioClickIncidence
			)
		}

		if (oldSettings.depthAudioClickIncidence != settings.depthAudioClickIncidence) {
			uniffi.NativeLib.setAudioSettings(
				settings.depthAudioFrequency.toFloat(),
				settings.depthAudioClickIncidence
			)
		}

		val enableNpuChanged = oldSettings.enableNpu != settings.enableNpu

		CoroutineScope(loadAIModelExecutor.asCoroutineDispatcher()).launch {
			if (oldSettings.nlpModel != settings.nlpModel) {
				switchNlpModel(settings.nlpModel)
			}

			if (oldSettings.depthModel != settings.depthModel || enableNpuChanged) {
				Log.d(UI_LOG_TAG, "[EyeAIApp.updateSettings] DepthModel is set to ${settings.depthModel}")
				Log.d(UI_LOG_TAG, "[EyeAIApp.updateSettings] Enable NPU is set to ${settings.enableNpu}")
				switchDepthModel(settings.depthModel)
			}

			if (oldSettings.enableSpeechRecognition != settings.enableSpeechRecognition) {
				Log.d(UI_LOG_TAG, "[EyeAIApp.updateSettings] EnableSpeechRecognition is set to ${settings.enableSpeechRecognition}")
				if (!settings.enableSpeechRecognition) {
					Log.d(UI_LOG_TAG, "[EyeAIApp.updateSettings] Closing Vosk service")
					voskModel.closeService()
				}
			}

			if (oldSettings.enableObjectDetection != settings.enableObjectDetection || enableNpuChanged) {
				Log.d(UI_LOG_TAG, "[EyeAIApp.updateSettings] EnableObjectDetection is set to ${settings.enableObjectDetection}")
				Log.d(UI_LOG_TAG, "[EyeAIApp.updateSettings] Enable NPU is set to ${settings.enableNpu}")
				if (settings.enableObjectDetection) {
					yoloModel.create(baseContext, npuQnnDelegateDirectory!!, settings.enableNpu)
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
		if (metricDepthModel?.name == modelName && metricDepthModel?.enableNpu == settings.enableNpu) return

		metricDepthModel?.close()
		metricDepthModel = null

		metricDepthModel = findDepthModelInfo(modelName)
			.createDepthModel(this, npuQnnDelegateDirectory!!, settings.enableNpu)
	}

	private fun switchNlpModel(modelId: String) {
		val modelInfo = NLPModelInfo.findById(modelId)
		if (nlpModel.info.id == modelInfo.id && nlpModel.isInitialized) return
		nlpModel.create(this, modelInfo)
	}

	private fun findDepthModelInfo(modelName: String): MetricDepthModelInfo {
		return DEPTH_MODELS.find { it.name == modelName }
			?: (DEPTH_MODELS.find { it.name == DEFAULT_DEPTH_MODEL_NAME } ?: DEPTH_MODELS[0])
	}
}

fun getLastAppUpdateTime(context: Context): Long {
	try {
		val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
		return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
			packageInfo.lastUpdateTime
		} else {
			// Fallback
			File(context.packageCodePath).lastModified()
		}
	} catch (e: PackageManager.NameNotFoundException) {
		e.printStackTrace()
		return 0L
	}
}
