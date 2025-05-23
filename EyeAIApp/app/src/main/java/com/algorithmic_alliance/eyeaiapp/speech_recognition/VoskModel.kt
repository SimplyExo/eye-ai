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
import org.vosk.android.SpeechStreamService
import org.vosk.android.StorageService

class VoskModel(val context: Context, val modelName: String) : AutoCloseable {
	private var model: Model? = null
	private var speechService: SpeechService? = null
	private var speechStreamService: SpeechStreamService? = null

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

	fun init(
		onPartialResult: (partial: String) -> Unit,
		onFinalResult: (final: String) -> Unit
	) {
		this.onPartialResultCallback = onPartialResult
		this.onFinalResultCallback = onFinalResult

		LibVosk.setLogLevel(LogLevel.INFO)

		StorageService.unpack(
			context, modelName, "unpacked_vosk_model",
			{ model ->
				this.model = model
				try {
					val rec = Recognizer(model, 16000.0f)
					speechService = SpeechService(rec, 16000.0f)
					speechService!!.startListening(recognitionListener)
				} catch (e: Exception) {
					Log.e(
						EyeAIApp.APP_LOG_TAG,
						"[VoskModel] failed to create speech recognition listener for model: $modelName: $e"
					)
				}
			},
			{ exception ->
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"[VoskModel] Failed to unpack Vosk speech recognition model '${modelName}': $exception"
				)
			}
		)
	}

	override fun close() {
		speechService?.let {
			it.stop()
			it.shutdown()
		}

		speechStreamService?.stop()
	}

	fun setPaused(paused: Boolean) {
		speechService?.apply { setPause(paused) }
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