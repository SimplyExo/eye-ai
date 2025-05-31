package com.algorithmic_alliance.eyeaiapp.speech_recognition

import android.content.Context
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import org.json.JSONObject
import org.vosk.LibVosk
import org.vosk.LogLevel
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.vosk.android.StorageService

class VoskModel(val context: Context, val modelName: String) {
	companion object {
		private const val SAMPLE_RATE = 48000.0f // 16000.0f
	}

	private var model: Model? = null
	private var speechService: SpeechService? = null

	private var onPartialResultCallback: (partial: String) -> Unit = {}
	private var onFinalResultCallback: (partial: String) -> Unit = {}

	private var recognitionListener = object : RecognitionListener {
		override fun onPartialResult(hypothesis: String) {
			parsePartialOutput(hypothesis)?.let {
				onPartialResultCallback(it)
			} ?: run {
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"[VoskModel] failed to parse partial result json format"
				)
			}
		}

		override fun onResult(hypothesis: String) {
			parseResultOutput(hypothesis)?.let {
				onFinalResultCallback(it)
			} ?: run {
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"[VoskModel] failed to parse result json format"
				)
			}
		}

		override fun onFinalResult(hypothesis: String) {
			parseResultOutput(hypothesis)?.let {
				onFinalResultCallback(it)
			} ?: run {
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"[VoskModel] failed to parse final result json format"
				)
			}
		}

		override fun onError(exception: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] $exception")
		}

		override fun onTimeout() {
			Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] timeout")
		}
	}

	init {
		LibVosk.setLogLevel(LogLevel.INFO)
	}

	fun isInitialized(): Boolean {
		return model != null && speechService != null
	}

	fun initService(
		onPartialResult: (partial: String) -> Unit,
		onFinalResult: (final: String) -> Unit,
		onModelLoaded: () -> Unit
	) {
		this.onPartialResultCallback = onPartialResult
		this.onFinalResultCallback = onFinalResult

		if (model != null && speechService != null)
			return

		StorageService.unpack(
			context, modelName, "unpacked_vosk_model",
			{ model ->
				this.model = model
				startListening()
				onModelLoaded()
			},
			{ exception ->
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"[VoskModel] Failed to unpack Vosk speech recognition model '${modelName}': $exception"
				)
			}
		)
	}

	fun closeService() {
		speechService?.apply {
			stop()
			shutdown()
		}
	}

	fun startListening() {
		if (model == null) {
			Log.w(EyeAIApp.APP_LOG_TAG, "[VoskModel] cannot startListening: model not loaded")
			return
		}

		if (speechService == null) {
			try {
				val rec = Recognizer(model, SAMPLE_RATE)
				speechService = SpeechService(rec, SAMPLE_RATE)
			} catch (e: Exception) {
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"[VoskModel] failed to create speech recognition listener for model: $modelName: $e"
				)
			}
		}
		speechService?.startListening(recognitionListener)
	}

	fun stopListening() {
		speechService?.stop()
		speechService = null
	}

	/**
	 * parses the partial output from the json output of vosk
	 * @return null, if [outputJson] does not contain "partial" key */
	private fun parsePartialOutput(outputJson: String): String? {
		try {
			val jsonObject = JSONObject(outputJson)
			return jsonObject.getString("partial")
		} catch (e: Exception) {
			e.printStackTrace()
			return null
		}
	}

	/**
	 * parses the result output from the json output of vosk
	 * @return null, if [outputJson] does not contain "text" key */
	private fun parseResultOutput(outputJson: String): String? {
		try {
			val jsonObject = JSONObject(outputJson)
			return jsonObject.getString("text")
		} catch (e: Exception) {
			e.printStackTrace()
			return null
		}
	}
}