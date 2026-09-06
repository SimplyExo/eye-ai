package com.algorithmic_alliance.eyeaiapp

import android.content.Context
import androidx.preference.PreferenceManager
import androidx.core.content.edit
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModelInfo

data class Settings(
	var depthModel: String,
	/** Optional MiDaS limit; null deliberately enables unbounded benchmark operation. */
	var maxDepthFrameRate: Int?,
	var showProfilingInfo: Boolean,
	var showDebugInputBitmap: Boolean,
	var enableSpeechRecognition: Boolean,
	var nlpModel: String,
	var enableObjectDetection: Boolean,
	/** Hard maximum object-detection budget in FPS; adaptive modes run below it. */
	var maxObjectDetectionFrameRate: Int?,
	var enableOCR: Boolean,
	val inputSource: String?,
	val mediaSource: String?,
	val eyeAIVisionIP: String?,
	var depthAudioPlayback: Boolean,
	var objectAudioPlayback: Boolean,	
	var depthAudioFrequency: Int,
	var depthAudioClickIncidence: Int,
	var objectAudioPlaybackLanguage: String?,
	var enableNpu: Boolean,
	var jpegCompression: Int

) : Cloneable {
	companion object {
		const val DEFAULT_FRAME_RATE_LIMIT: Int = 30
		const val MIN_DEPTH_FRAME_RATE: Int = 1
		const val MAX_DEPTH_FRAME_RATE: Int = 60
		const val MIN_OBJECT_DETECTION_FRAME_RATE: Int = 5
		const val MAX_OBJECT_DETECTION_FRAME_RATE: Int = 60

		/** Keep enabled MiDaS limiter settings inside the 1..60 FPS slider range. */
		fun normalizeDepthFrameRate(value: Int): Int =
			value.coerceIn(MIN_DEPTH_FRAME_RATE, MAX_DEPTH_FRAME_RATE)

		/** Preserve null as the explicit unbounded test mode. */
		fun effectiveDepthFrameRate(value: Int?): Int? =
			value?.let(::normalizeDepthFrameRate)

		/** Keep enabled Object-Detection budgets inside the 5..60 FPS slider range. */
		fun normalizeObjectDetectionFrameRate(value: Int): Int =
			value.coerceIn(MIN_OBJECT_DETECTION_FRAME_RATE, MAX_OBJECT_DETECTION_FRAME_RATE)

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
				normalizeDepthFrameRate(sharedPreferences.getInt(
					context.getString(R.string.max_depth_frame_rate_setting),
					DEFAULT_FRAME_RATE_LIMIT
				))
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

			val configuredNlpModel = sharedPreferences.getString(
				context.getString(R.string.nlp_model_setting),
				NLPModelInfo.DEFAULT_MODEL_ID
			).toString()
			val nlpModel = NLPModelInfo.findById(configuredNlpModel).id

			val enableObjectDetection = sharedPreferences.getBoolean(
				context.getString(R.string.enable_object_detection_setting),
				true
			)

			val objectDetectionFrameRateLimitEnabled = sharedPreferences.getBoolean(
				context.getString(R.string.enable_object_detection_frame_rate_limit_setting),
				true
			)

			val maxObjectDetectionFrameRate = if (objectDetectionFrameRateLimitEnabled) {
				normalizeObjectDetectionFrameRate(sharedPreferences.getInt(
					context.getString(R.string.max_object_detection_frame_rate_setting),
					DEFAULT_FRAME_RATE_LIMIT
				))
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

			val jpegCompression = sharedPreferences.getInt(context.getString(R.string.jpeg_compression),
				15
			)

			val depthAudioClickIncidence = sharedPreferences.getInt(context.getString(R.string.audio_playback_rate_setting), 2)

			val depthAudioFrequency = sharedPreferences.getInt(context.getString(R.string.audio_frequency_range_setting), 500)

			val objectAudioPlaybackLanguage = sharedPreferences.getString(context.getString(R.string.object_playback_language), "english")

			val enableNpu = sharedPreferences.getBoolean(
				context.getString(R.string.enable_npu_delegate_setting),
				true
			)

			return Settings(
				depthModel,
				maxDepthFrameRate,
				showProfilingInfo,
				showDebugInputBitmap,
				enableSpeechRecognition,
				nlpModel,
				enableObjectDetection,
				maxObjectDetectionFrameRate,
				enableOCR,
				inputSource,
				mediaSource,
				eyeAIVisionIP,
				depthAudioPlayback,
				objectAudioPlayback,
				depthAudioFrequency,
				depthAudioClickIncidence,
				objectAudioPlaybackLanguage,
				enableNpu,
				jpegCompression
			)
		}
	}

	public override fun clone(): Settings = Settings(
		depthModel,
		maxDepthFrameRate,
		showProfilingInfo,
		showDebugInputBitmap,
		enableSpeechRecognition,
		nlpModel,
		enableObjectDetection,
		maxObjectDetectionFrameRate,
		enableOCR,
		inputSource,
		mediaSource,
		eyeAIVisionIP,
		depthAudioPlayback,
		objectAudioPlayback,
		depthAudioFrequency,
		depthAudioClickIncidence,
		objectAudioPlaybackLanguage,
		enableNpu,
		jpegCompression
	)

	fun save(context: Context) {
		val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
		sharedPreferences.edit {

			// Audiosettings that can be modified via speech
			putInt("audio_playback_rate", depthAudioClickIncidence)
			putInt("audio_frequency_range", depthAudioFrequency)
			putString("object_playback_language", objectAudioPlaybackLanguage)

			//other settings
			putString(context.getString(R.string.depth_model_setting), depthModel)
			putBoolean(context.getString(R.string.show_profiling_info_setting), showProfilingInfo)
			putBoolean(
				context.getString(R.string.show_debug_input_bitmap_setting),
				showDebugInputBitmap
			)
			putBoolean(
				context.getString(R.string.enable_speech_recognition_setting),
				enableSpeechRecognition
			)
			putString(context.getString(R.string.nlp_model_setting), nlpModel)
			putBoolean(
				context.getString(R.string.enable_object_detection_setting),
				enableObjectDetection
			)
			putBoolean(context.getString(R.string.enable_ocr_setting), enableOCR)
			putBoolean(context.getString(R.string.depth_playback_setting), depthAudioPlayback)
			putBoolean(context.getString(R.string.object_playback_setting), objectAudioPlayback)
			putBoolean(context.getString(R.string.enable_npu_delegate_setting), enableNpu)
			putInt(context.getString(R.string.jpeg_compression), jpegCompression)

			// Frame Rate Limits
			maxDepthFrameRate?.let {
				putBoolean(context.getString(R.string.enable_depth_frame_rate_limit_setting), true)
				putInt(
					context.getString(R.string.max_depth_frame_rate_setting),
					normalizeDepthFrameRate(it),
				)
			} ?: putBoolean(
				context.getString(R.string.enable_depth_frame_rate_limit_setting),
				false
			)

			maxObjectDetectionFrameRate?.let {
				putBoolean(
					context.getString(R.string.enable_object_detection_frame_rate_limit_setting),
					true
				)
				putInt(
					context.getString(R.string.max_object_detection_frame_rate_setting),
					normalizeObjectDetectionFrameRate(it),
				)
			} ?: putBoolean(
				context.getString(R.string.enable_object_detection_frame_rate_limit_setting),
				false
			)

		}
	}
}
