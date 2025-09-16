package com.algorithmic_alliance.eyeaiapp

import android.content.Context
import androidx.preference.PreferenceManager

data class Settings(
	var depthModel: String,
	var maxDepthFrameRate: Int?,
	var showProfilingInfo: Boolean,
	var showDebugInputBitmap: Boolean,
	var enableSpeechRecognition: Boolean,
	var googleAiStudioApiKey: String?,
	var customGoogleGenAIStudioEndpoint: String?,
	var enableObjectDetection: Boolean,
	var maxObjectDetectionFrameRate: Int?,
	var enableOCR: Boolean,
	val inputSource: String?,
	val mediaSource: String?,
	val eyeAIVisionIP: String?,
	var depthAudioPlayback: Boolean,
	var objectAudioPlayback: Boolean
) : Cloneable {
	companion object {
		const val DEFAULT_FRAME_RATE_LIMIT: Int = 30

		fun load(context: Context): Settings {
			val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)

			val depthModel = sharedPreferences.getString(
				context.getString(R.string.depth_model_setting),
				EyeAIApp.DEFAULT_DEPTH_MODEL_NAME
			).toString()

			val depthFrameRateLimitEnabled = sharedPreferences.getBoolean(
				context.getString(R.string.enable_depth_frame_rate_limit_setting),
				true
			)

			val maxDepthFrameRate = if (depthFrameRateLimitEnabled) {
				sharedPreferences.getString(
					context.getString(R.string.max_depth_frame_rate_setting),
					DEFAULT_FRAME_RATE_LIMIT.toString()
				)
					?.toIntOrNull()
			} else {
				null
			}

			val showProfilingInfo = sharedPreferences.getBoolean(
				context.getString(R.string.show_profiling_info_setting),
				false
			)

			val showDebugInputBitmap = sharedPreferences.getBoolean(
				context.getString(R.string.show_debug_input_bitmap_setting),
				false
			)

			val enableSpeechRecognition = sharedPreferences.getBoolean(
				context.getString(R.string.enable_speech_recognition_setting),
				true
			)

			val googleAiStudioApiKey = sharedPreferences.getString(
				context.getString(R.string.google_ai_studio_api_key_stetting),
				null
			)

			val customGoogleGenAIStudioEndpoint = sharedPreferences.getString(
				context.getString(R.string.custom_google_gen_ai_studio_endpoint_setting),
				null
			)

			val enableObjectDetection = sharedPreferences.getBoolean(
				context.getString(R.string.enable_object_detection_setting),
				true
			)

			val objectDetectionFrameRateLimitEnabled = sharedPreferences.getBoolean(
				context.getString(R.string.enable_object_detection_frame_rate_limit_setting),
				true
			)

			val maxObjectDetectionFrameRate = if (objectDetectionFrameRateLimitEnabled) {
				sharedPreferences.getString(
					context.getString(R.string.max_object_detection_frame_rate_setting),
					DEFAULT_FRAME_RATE_LIMIT.toString()
				)?.toIntOrNull()
			} else {
				null
			}

			val enableOCR = sharedPreferences.getBoolean(
				context.getString(R.string.enable_ocr_setting),
				true
			)

			val inputSource = sharedPreferences.getString(
				context.getString(R.string.input_source_setting),
				context.getString(R.string.input_is_camera)
			)

			val mediaSource = sharedPreferences.getString(
				context.getString(R.string.media_path_setting),
				""
			)

			val eyeAIVisionIP = sharedPreferences.getString(
				context.getString(R.string.eyeaivision_ip_setting),
				""
			)

			val depthAudioPlayback = sharedPreferences.getBoolean(
				context.getString(R.string.depth_playback_setting),
				true
			)

			val objectAudioPlayback = sharedPreferences.getBoolean(
				context.getString(R.string.object_playback_setting),
				true
			)

			return Settings(
				depthModel,
				maxDepthFrameRate,
				showProfilingInfo,
				showDebugInputBitmap,
				enableSpeechRecognition,
				googleAiStudioApiKey,
				customGoogleGenAIStudioEndpoint,
				enableObjectDetection,
				maxObjectDetectionFrameRate,
				enableOCR,
				inputSource,
				mediaSource,
				eyeAIVisionIP,
				depthAudioPlayback,
				objectAudioPlayback
			)
		}
	}

	public override fun clone(): Settings = Settings(
		depthModel,
		maxDepthFrameRate,
		showProfilingInfo,
		showDebugInputBitmap,
		enableSpeechRecognition,
		googleAiStudioApiKey,
		customGoogleGenAIStudioEndpoint,
		enableObjectDetection,
		maxObjectDetectionFrameRate,
		enableOCR,
		inputSource,
		mediaSource,
		eyeAIVisionIP,
		depthAudioPlayback,
		objectAudioPlayback
	)
}