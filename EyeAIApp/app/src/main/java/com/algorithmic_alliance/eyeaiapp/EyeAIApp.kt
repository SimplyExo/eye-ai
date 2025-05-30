package com.algorithmic_alliance.eyeaiapp

import android.app.Application
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.depth.DepthModel
import com.algorithmic_alliance.eyeaiapp.depth.DepthModelInfo
import com.algorithmic_alliance.eyeaiapp.speech_recognition.VoskModel

/**
 * App class that holds everything that should persist when switching to another app, for example
 * the camera handle and the loaded depth model
 */
class EyeAIApp : Application() {
	var cameraManager = CameraManager()
	lateinit var settings: Settings
		private set
	lateinit var depthModel: DepthModel
		private set
	lateinit var voskModel: VoskModel
		private set

	companion object {
		const val APP_LOG_TAG = "Eye AI"

		const val DEFAULT_DEPTH_MODEL_NAME = "MiDaS V2.1"

		/** make sure to change res/values/arrays.xml 'depth_models' array as well! */
		val DEPTH_MODELS =
			arrayOf(
				DepthModelInfo(
					DEFAULT_DEPTH_MODEL_NAME,
					"midas_v2_1_256x256.tflite",
					256,
					floatArrayOf(123.675f, 116.28f, 103.53f),
					floatArrayOf(58.395f, 57.12f, 57.375f)
				),
				DepthModelInfo(
					"MiDaS V2.1 (quantized)",
					"midas_v2_1_256x256_quantized.tflite",
					256,
					floatArrayOf(123.675f, 116.28f, 103.53f),
					floatArrayOf(58.395f, 57.12f, 57.375f)
				),
				DepthModelInfo(
					"Depth Anything V2",
					"depth_anything_v2_vits_210x210.onnx",
					210,
					floatArrayOf(0.485f, 0.456f, 0.406f),
					floatArrayOf(0.229f, 0.224f, 0.225f)
				)
			)
	}

	override fun onCreate() {
		super.onCreate()

		settings = Settings(this)

		depthModel =
			findDepthModelInfo(settings.depthModel)
				.createDepthModel(this, settings.profilingEnabled)!!

		voskModel = VoskModel(this, "model-de")
	}

	fun updateSettings() {
		val newSettings = Settings(this)

		if (settings.depthModel != newSettings.depthModel ||
			settings.profilingEnabled != newSettings.profilingEnabled
		) {
			switchDepthModel(newSettings.depthModel, newSettings.profilingEnabled)
		}

		settings = newSettings
	}

	private fun switchDepthModel(modelName: String, profilingEnabled: Boolean) {
		val newDepthModel = findDepthModelInfo(modelName).createDepthModel(this, profilingEnabled)
		if (newDepthModel != null) depthModel = newDepthModel
		else
			Log.e(
				APP_LOG_TAG,
				"Failed to switch from model ${depthModel.getName()} to new model $modelName"
			)
	}

	private fun findDepthModelInfo(modelName: String): DepthModelInfo {
		return DEPTH_MODELS.find { it.name == modelName }
			?: (DEPTH_MODELS.find { it.name == DEFAULT_DEPTH_MODEL_NAME } ?: DEPTH_MODELS[0])
	}
}
