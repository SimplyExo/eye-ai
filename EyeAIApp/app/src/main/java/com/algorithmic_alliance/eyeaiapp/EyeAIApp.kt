package com.algorithmic_alliance.eyeaiapp

import android.app.Application
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.depth.DepthModel
import com.algorithmic_alliance.eyeaiapp.depth.DepthModelInfo

/**
 * App class that holds everything that should persist when switching to another app,
 * for example the camera handle and the loaded depth model
 */
class EyeAIApp : Application() {
	var cameraManager = CameraManager()
	var selectedModelIndex = 0
		private set
	lateinit var depthModel: DepthModel

	companion object {
		const val APP_LOG_TAG = "Eye AI"

		val MODELS = arrayOf(
			DepthModelInfo(
				"MiDaS V2.1",
				"midas_v2_1_256x256.tflite",
				false,
				256,
				floatArrayOf(123.675f, 116.28f, 103.53f),
				floatArrayOf(58.395f, 57.12f, 57.375f)
			),
			DepthModelInfo(
				"MiDaS V2.1 (quantized)",
				"midas_v2_1_256x256_quantized.tflite",
				true,
				256,
				floatArrayOf(123.675f, 116.28f, 103.53f),
				floatArrayOf(58.395f, 57.12f, 57.375f)
			),
			DepthModelInfo(
				"Depth Anything V2",
				"depth_anything_v2_vits_210x210.onnx",
				false,
				210,
				floatArrayOf(0.485f, 0.456f, 0.406f),
				floatArrayOf(0.229f, 0.224f, 0.225f)
			)
		)
	}

	override fun onCreate() {
		super.onCreate()

		depthModel = MODELS[selectedModelIndex].createDepthModel(this)!!
	}

	fun switchModel(newModelIndex: Int) {
		selectedModelIndex = newModelIndex
		val newDepthModel = MODELS[selectedModelIndex].createDepthModel(this)
		if (newDepthModel != null)
			depthModel = newDepthModel
		else
			Log.e(
				APP_LOG_TAG,
				"Failed to switch from model ${depthModel.getName()} to new model ${MODELS[newModelIndex].name}"
			)
	}
}
