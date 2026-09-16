package com.algorithmic_alliance.eyeaiapp

import android.app.Activity
import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.annotation.RequiresApi
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModel
import com.algorithmic_alliance.eyeaiapp.depth.MetricDepthModelInfo
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModel
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntime
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntimeService
import com.algorithmic_alliance.eyeaiapp.settingsparser.LocalSettingsParser
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger

/**
 * Process owner for EyeAIRuntime. The runtime is not owned by MainActivity on purpose.
 * The foreground service controls only its active operation lifecycle.
 */
class EyeAIApp : Application() {
	private val visibleActivityCount = AtomicInteger(0)

	@Volatile
	lateinit var settings: Settings
		private set

	lateinit var runtime: EyeAIRuntime
		private set

	val nlpModel: NLPModel
		get() = runtime.nlpModel
	val aiData = AIModelData

	// Loaded lazily and shared across short-lived StateMachine instances.
	val confirmationModel: ConfirmationModel by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
		val started = System.nanoTime()
		Log.i(
			APP_LOG_TAG,
			"[DecisionTrace][ConfirmationModel][LOAD] outcome=STARTED " + "model=${ConfirmationModel.MODEL_ID} asset=${ConfirmationModel.ASSET_PATH} " + "execution=LOCAL apiCalled=false",
		)
		try {
			ConfirmationModel.fromAssets(this).also { model ->
				Log.i(
					APP_LOG_TAG,
					"[DecisionTrace][ConfirmationModel][LOAD] outcome=SUCCESS " + "model=${ConfirmationModel.MODEL_ID} featureCount=${model.featureCount} " + "threshold=${
						String.format(
							Locale.US, "%.4f", model.confidenceThreshold
						)
					} " + "duration=${(System.nanoTime() - started) / 1_000_000}ms " + "execution=LOCAL apiCalled=false",
				)
			}
		} catch (error: Throwable) {
			Log.e(
				APP_LOG_TAG,
				"[DecisionTrace][ConfirmationModel][LOAD] outcome=FAILED " + "model=${ConfirmationModel.MODEL_ID} " + "duration=${(System.nanoTime() - started) / 1_000_000}ms " + "execution=LOCAL apiCalled=false",
				error,
			)
			throw error
		}
	}

	// lazy and runtime-owned.
	private val localSettingsParserLazy = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
		val started = System.nanoTime()
		Log.i(
			APP_LOG_TAG,
			"[DecisionTrace][SettingsParser][LOAD] architecture=SPECIALIZED_WORD_OPERATION_CHAR_SPEAKER " + "execution=LOCAL apiCalled=false outcome=STARTED",
		)
		try {
			LocalSettingsParser.fromAssets(this).also {
				Log.i(
					APP_LOG_TAG,
					"[DecisionTrace][SettingsParser][LOAD] execution=LOCAL apiCalled=false " + "outcome=SUCCESS duration=${(System.nanoTime() - started) / 1_000_000}ms",
				)
			}
		} catch (error: Throwable) {
			Log.e(
				APP_LOG_TAG,
				"[DecisionTrace][SettingsParser][LOAD] execution=LOCAL apiCalled=false outcome=FAILED " + "duration=${(System.nanoTime() - started) / 1_000_000}ms",
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
		Log.e(APP_LOG_TAG, "!!! EYEAI APP ONCREATE !!!")
		super.onCreate()
		registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
			override fun onActivityStarted(activity: Activity) {
				visibleActivityCount.incrementAndGet()
				EyeAIRuntimeService.onUiVisibilityChanged(this@EyeAIApp, isVisible = true)
			}

			override fun onActivityStopped(activity: Activity) {
				visibleActivityCount.updateAndGet { count -> (count - 1).coerceAtLeast(0) }
				EyeAIRuntimeService.onUiVisibilityChanged(
					this@EyeAIApp,
					isVisible = hasVisibleActivity(),
				)
			}

			override fun onActivityCreated(activity: Activity, state: Bundle?) = Unit
			override fun onActivityResumed(activity: Activity) = Unit
			override fun onActivityPaused(activity: Activity) = Unit
			override fun onActivitySaveInstanceState(activity: Activity, state: Bundle) = Unit
			override fun onActivityDestroyed(activity: Activity) = Unit
		})
		uniffi.NativeLib.initAndroidLogging()
		settings = Settings.load(this)
		runtime = EyeAIRuntime(this)
		runtime.initializeModels()
	}

	// True while an EyeAI Activity is visible, including the settings screen.
	// This is used to prevent the user from killing core functionality while using the app
	internal fun hasVisibleActivity(): Boolean = visibleActivityCount.get() > 0

	@RequiresApi(Build.VERSION_CODES.P)
	override fun onTerminate() {
		if (::runtime.isInitialized) runtime.close()
		super.onTerminate()
	}

	fun updateSettings() {
		val oldSettings = settings.clone()
		settings = Settings.load(this)
		runtime.onSettingsChanged(oldSettings)
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
