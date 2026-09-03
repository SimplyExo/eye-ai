package com.algorithmic_alliance.eyeaiapp

import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.util.Size
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModel
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModel
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModelInfo
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModel
import com.algorithmic_alliance.eyeaiapp.object_detection.YoloModel
import com.algorithmic_alliance.eyeaiapp.ocr.GoogleOCR
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntime
import com.algorithmic_alliance.eyeaiapp.settingsparser.LocalSettingsParser
import com.algorithmic_alliance.eyeaiapp.speech_recognition.VoskModel
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import java.io.File
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Process owner for the one [EyeAIRuntime]. The runtime is deliberately not
 * owned by MainActivity; the foreground service controls only its active
 * operation lifecycle.
 */
class EyeAIApp : Application() {
    @Volatile
    lateinit var settings: Settings
        private set

    lateinit var runtime: EyeAIRuntime
        private set

    /** Compatibility accessors for existing model/state-machine code. */
    val speechThreadExecutor: ExecutorService
        get() = runtime.speechThreadExecutorForStateMachine
    var lastDialogContext: String?
        get() = runtime.lastDialogContext
        set(value) {
            // StateMachine compatibility only; ownership remains in runtime.
            runtime.setLastDialogContextFromCompatibility(value)
        }
    val metricDepthModel: MetricDepthModel?
        get() = runtime.metricDepthModel
    val voskModel: VoskModel
        get() = runtime.voskModel
    val yoloModel: YoloModel
        get() = runtime.yoloModel
    val nlpModel: NLPModel
        get() = runtime.nlpModel
    val ocrModel: GoogleOCR
        get() = runtime.ocrModel
    val textToSpeechInstance: TextToSpeechInstance
        get() = runtime.textToSpeechInstance
    val cameraManager
        get() = runtime.cameraManager
    val voskUserStart: AtomicBoolean
        get() = runtime.voskUserStart
    val aiData = AIModelData
    val npuQnnDelegateDirectory: String
        get() = runtime.npuQnnDelegateDirectory

    var lastLlmJsonResponse: String? = null

    /** Loaded lazily and shared across short-lived StateMachine instances. */
    val confirmationModel: ConfirmationModel by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        val started = System.nanoTime()
        Log.i(
            APP_LOG_TAG,
            "[DecisionTrace][ConfirmationModel][LOAD] outcome=STARTED " +
                "model=${ConfirmationModel.MODEL_ID} asset=${ConfirmationModel.ASSET_PATH} " +
                "execution=LOCAL apiCalled=false",
        )
        try {
            ConfirmationModel.fromAssets(this).also { model ->
                Log.i(
                    APP_LOG_TAG,
                    "[DecisionTrace][ConfirmationModel][LOAD] outcome=SUCCESS " +
                        "model=${ConfirmationModel.MODEL_ID} featureCount=${model.featureCount} " +
                        "threshold=${String.format(Locale.US, "%.4f", model.confidenceThreshold)} " +
                        "duration=${(System.nanoTime() - started) / 1_000_000}ms " +
                        "execution=LOCAL apiCalled=false",
                )
            }
        } catch (error: Throwable) {
            Log.e(
                APP_LOG_TAG,
                "[DecisionTrace][ConfirmationModel][LOAD] outcome=FAILED " +
                    "model=${ConfirmationModel.MODEL_ID} " +
                    "duration=${(System.nanoTime() - started) / 1_000_000}ms " +
                    "execution=LOCAL apiCalled=false",
                error,
            )
            throw error
        }
    }

    /** The frozen local settings parsers remain lazy and runtime-owned. */
    private val localSettingsParserLazy = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        val started = System.nanoTime()
        Log.i(
            APP_LOG_TAG,
            "[DecisionTrace][SettingsParser][LOAD] architecture=SPECIALIZED_WORD_OPERATION_CHAR_SPEAKER " +
                "execution=LOCAL apiCalled=false outcome=STARTED",
        )
        try {
            LocalSettingsParser.fromAssets(this).also {
                Log.i(
                    APP_LOG_TAG,
                    "[DecisionTrace][SettingsParser][LOAD] execution=LOCAL apiCalled=false " +
                        "outcome=SUCCESS duration=${(System.nanoTime() - started) / 1_000_000}ms",
                )
            }
        } catch (error: Throwable) {
            Log.e(
                APP_LOG_TAG,
                "[DecisionTrace][SettingsParser][LOAD] execution=LOCAL apiCalled=false outcome=FAILED " +
                    "duration=${(System.nanoTime() - started) / 1_000_000}ms",
                error,
            )
            throw error
        }
    }

    val localSettingsParser: LocalSettingsParser
        get() = localSettingsParserLazy.value

    internal fun localSettingsParserLazyIsInitialized(): Boolean =
        localSettingsParserLazy.isInitialized()

    companion object {
        const val APP_LOG_TAG = "Eye AI"
        const val DEFAULT_DEPTH_MODEL_NAME = "MiDaS V2.1"

        val DEPTH_MODELS = arrayOf(
            MetricDepthModelInfo(DEFAULT_DEPTH_MODEL_NAME, "midas_v2_1_256x256.tflite"),
            MetricDepthModelInfo(
                "MiDaS V2.1 (quantized)",
                "midas_v2_1_256x256_quantized.tflite",
            ),
        )

        val PREFERRED_CAMERA_RESOLUTION = Size(640, 640)
    }

    override fun onCreate() {
        super.onCreate()
        uniffi.NativeLib.initAndroidLogging()
        settings = Settings.load(this)
        runtime = EyeAIRuntime(this)
        runtime.initializeModels()
    }

    override fun onTerminate() {
        if (::runtime.isInitialized) runtime.close()
        super.onTerminate()
    }

    fun updateSettings() {
        val oldSettings = settings.clone()
        settings = Settings.load(this)
        runtime.onSettingsChanged(oldSettings)
    }

    /** Keeps the legacy App-level setter source-compatible without moving ownership back to UI. */
    internal fun setLastDialogContextFromRuntime(value: String?) {
        runtime.setLastDialogContextFromCompatibility(value)
    }
}

fun getLastAppUpdateTime(context: Context): Long {
    return try {
        val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            packageInfo.lastUpdateTime
        } else {
            File(context.packageCodePath).lastModified()
        }
    } catch (_: PackageManager.NameNotFoundException) {
        0L
    }
}
