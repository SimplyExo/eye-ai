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
		private const val SAMPLE_RATE = 48000.0f
	}

	private var model: Model? = null
	private var speechService: SpeechService? = null
	private var isListening = false

	private var onPartialResultCallback: (partial: String) -> Unit = {}
	private var onFinalResultCallback: (partial: String) -> Unit = {}


	fun isListening(): Boolean = isListening
	private val recognitionListener = object : RecognitionListener {
		override fun onPartialResult(hypothesis: String) {
			parsePartialOutput(hypothesis)?.let {
				onPartialResultCallback(it)
			} ?: run {
				Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] failed to parse partial result json format")
			}
		}

		override fun onResult(hypothesis: String) {
			parseResultOutput(hypothesis)?.let {
				onFinalResultCallback(it)
			} ?: run {
				Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] failed to parse result json format")
			}
		}

		override fun onFinalResult(hypothesis: String) {
			parseResultOutput(hypothesis)?.let {
				onFinalResultCallback(it)
			} ?: run {
				Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] failed to parse final result json format")
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
		LibVosk.setLogLevel(LogLevel.DEBUG)
	}

	fun initService(
		onPartialResult: (partial: String) -> Unit,
		onFinalResult: (final: String) -> Unit,
		onModelLoaded: () -> Unit
	) {
		this.onPartialResultCallback = onPartialResult
		this.onFinalResultCallback = onFinalResult

		if (model != null) {
			onModelLoaded()
			return
		}

		StorageService.unpack(
			context, modelName, "unpacked_vosk_model",
			{ loadedModel ->
				this.model = loadedModel
				Log.d(EyeAIApp.APP_LOG_TAG, "[VoskModel] model unpacked")
				onModelLoaded()
			},
			{ exception ->
				Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] Failed to unpack Vosk model '$modelName': $exception")
			}
		)
	}

	fun closeService() {
		try {
			stopListening()
			speechService?.shutdown()
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] Exception closing service: ", e)
		} finally {
			speechService = null
			model = null
			isListening = false
		}
	}

	@Synchronized
	fun startListening() {
		if (isListening) {
			Log.d(EyeAIApp.APP_LOG_TAG, "[VoskModel] startListening called but already listening; ignoring.")
			return
		}
		if (model == null) {
			Log.w(EyeAIApp.APP_LOG_TAG, "[VoskModel] cannot startListening: model not loaded")
			return
		}
		try {
			val rec = Recognizer(model, SAMPLE_RATE)
			speechService = SpeechService(rec, SAMPLE_RATE)
			speechService?.startListening(recognitionListener)
			isListening = true
			Log.d(EyeAIApp.APP_LOG_TAG, "[VoskModel] started listening")
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] failed to create/start speechService: $e")
			isListening = false
		}
	}

	@Synchronized
	fun stopListening() {
		if (!isListening) {
			Log.d(EyeAIApp.APP_LOG_TAG, "[VoskModel] stopListening called but not listening; ignoring.")
			return
		}
		try {
			speechService?.stop()
			Log.d(EyeAIApp.APP_LOG_TAG, "[VoskModel] stopped listening")
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] error while stopping speechService: ", e)
		} finally {
			isListening = false
		}
	}

	// ---------------------------
	// Result parsing helpers
	// ---------------------------
	private fun parsePartialOutput(outputJson: String): String? {
		return try {
			val jsonObject = JSONObject(outputJson)
			jsonObject.getString("partial")
		} catch (e: Exception) {
			e.printStackTrace()
			null
		}
	}

	private fun parseResultOutput(outputJson: String): String? {
		return try {
			val jsonObject = JSONObject(outputJson)
			jsonObject.getString("text")
		} catch (e: Exception) {
			e.printStackTrace()
			null
		}
	}
}