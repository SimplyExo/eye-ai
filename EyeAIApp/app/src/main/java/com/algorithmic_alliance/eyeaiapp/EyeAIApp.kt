package com.algorithmic_alliance.eyeaiapp

import android.app.Application
import android.content.Context
import android.util.Log
import android.util.Size
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.depth.DepthModel
import com.algorithmic_alliance.eyeaiapp.depth.DepthModelInfo
import com.algorithmic_alliance.eyeaiapp.speech_recognition.VoskModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * App class that holds everything that should persist when switching to another app, for example
 * the camera handle and the loaded depth model
 */
class EyeAIApp : Application() {
	var cameraManager = CameraManager()
	lateinit var settings: Settings
		private set
	var depthModel: DepthModel? = null
		private set
	var onDepthModelLoadedCallback: () -> Unit = {}

	/** can be [null] if enableSpeechRecognition is disabled in settings */
	var voskModel: VoskModel? = null
		private set

	companion object {
		const val APP_LOG_TAG = "Eye AI"

		const val DEFAULT_DEPTH_MODEL_NAME = "MiDaS V2.1"

		val DEPTH_MODELS =
			arrayOf(
				DepthModelInfo(
					DEFAULT_DEPTH_MODEL_NAME,
					"midas_v2_1_256x256.tflite",
					Size(256, 256)
				),
				DepthModelInfo(
					"MiDaS V2.1 (quantized)",
					"midas_v2_1_256x256_quantized.tflite",
					Size(256, 256)
				)
			)
	}

	override fun onCreate() {
		super.onCreate()

		settings = Settings(this)

		switchDepthModel(settings.depthModel)

		if (settings.enableSpeechRecognition)
			voskModel = VoskModel(this, "model-de")
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
