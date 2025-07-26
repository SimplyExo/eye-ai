package com.algorithmic_alliance.eyeaiapp

import android.content.Context
import androidx.preference.PreferenceManager

class Settings(val context: Context) {
	private var sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)

	var depthModel: String
		private set

	var showProfilingInfo: Boolean
		private set

	var showDebugInputBitmap: Boolean
		private set
	var enableSpeechRecognition: Boolean
		private set

	var googleAiStudioApiKey: String?

	var enableObjectDetection: Boolean
		private set

	var enableOCR: Boolean
		private set

	init {
		depthModel = sharedPreferences.getString(
			context.getString(R.string.depth_model_setting),
			EyeAIApp.DEFAULT_DEPTH_MODEL_NAME
		).toString()

		showProfilingInfo = sharedPreferences.getBoolean(
			context.getString(R.string.show_profiling_info_setting),
			false
		)

		showDebugInputBitmap = sharedPreferences.getBoolean(
			context.getString(R.string.show_debug_input_bitmap_setting),
			false
		)

		enableSpeechRecognition = sharedPreferences.getBoolean(
			context.getString(R.string.enable_speech_recognition_setting),
			true
		)

		googleAiStudioApiKey = sharedPreferences.getString(
			context.getString(R.string.google_ai_studio_api_key_stetting),
			null
		)

		enableObjectDetection = sharedPreferences.getBoolean(
			context.getString(R.string.enable_object_detection_setting),
			true
		)

		enableOCR = sharedPreferences.getBoolean(
			context.getString(R.string.enable_ocr_setting),
			true
		)
	}
}