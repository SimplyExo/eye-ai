package com.algorithmic_alliance.eyeaiapp.speech_recognition

import android.content.Context
import com.algorithmic_alliance.eyeaiapp.audio.AppSoundPlayer
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.R
import java.io.InputStream
import org.json.JSONObject
import org.vosk.LibVosk
import org.vosk.LogLevel
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.vosk.android.SpeechStreamService
import org.vosk.android.StorageService

class VoskModel(context: Context, val modelName: String) {
	private val context: Context = context.applicationContext

	companion object {
		private const val SAMPLE_RATE = 48000.0f
	}

	private var model: Model? = null
	private var speechService: SpeechService? = null
	private var speechStreamService: SpeechStreamService? = null

	@Volatile
	private var isListening = false

	@Volatile
	private var modelLoading = false
	private var modelLoadGeneration = 0L

	@Volatile
	private var onPartialResultCallback: (partial: String) -> Unit = {}

	@Volatile
	private var onFinalResultCallback: (partial: String) -> Unit = {}

	@Volatile
	private var onUpdateVoskUIStatus: (status: Boolean) -> Unit = {}

	private var activateSound: AppSoundPlayer? = null
	private var deactivateSound: AppSoundPlayer? = null


	fun isListening(): Boolean = isListening
	private val recognitionListener = object : RecognitionListener {
		override fun onPartialResult(hypothesis: String) {
			Log.v(EyeAIApp.APP_LOG_TAG, "[VoskModel] Received partial raw JSON: $hypothesis")
			parsePartialOutput(hypothesis)?.let { partial ->
				if (partial.isNotEmpty()) {
					Log.i(EyeAIApp.APP_LOG_TAG, "[VoskModel] Received partial hypothesis: '$partial'")
				}
				onPartialResultCallback(partial)
			} ?: run {
				Log.e(
					EyeAIApp.APP_LOG_TAG, "[VoskModel] failed to parse partial result json format"
				)
			}
		}

		override fun onResult(hypothesis: String) {
			Log.d(EyeAIApp.APP_LOG_TAG, "[VoskModel] Received result raw JSON: $hypothesis")
			parseResultOutput(hypothesis)?.let { text ->
				Log.i(EyeAIApp.APP_LOG_TAG, "[VoskModel] Received result hypothesis: '$text'")
				onFinalResultCallback(text)
			} ?: run {
				Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] failed to parse result json format")
			}
		}

		override fun onFinalResult(hypothesis: String) {
			Log.d(EyeAIApp.APP_LOG_TAG, "[VoskModel] Received final raw JSON: $hypothesis")
			parseResultOutput(hypothesis)?.let { final ->
				Log.i(EyeAIApp.APP_LOG_TAG, "[VoskModel] Received final hypothesis: '$final'")
				onFinalResultCallback(final)
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

		// MediaPlayer initialisieren
		activateSound = AppSoundPlayer.create(context, R.raw.activate)
		deactivateSound = AppSoundPlayer.create(context, R.raw.deactivate)
	}

	fun initService(
		onPartialResult: (partial: String) -> Unit,
		onFinalResult: (final: String) -> Unit,
		onModelLoaded: () -> Unit,
		onUpdateVoskUIStatus: (status: Boolean) -> Unit
	) {
		this.onPartialResultCallback = onPartialResult
		this.onFinalResultCallback = onFinalResult
		this.onUpdateVoskUIStatus = onUpdateVoskUIStatus
		ensureSoundPlayers()

		if (model != null) {
			onModelLoaded()
			return
		}

		if (modelLoading) return

		modelLoading = true
		val generation = ++modelLoadGeneration

		StorageService.unpack(context, modelName, "unpacked_vosk_model", { loadedModel ->
			val shouldNotify = synchronized(this) {
				if (generation != modelLoadGeneration) {
					false
				} else {
					this.model = loadedModel
					modelLoading = false
					true
				}
			}
			if (shouldNotify) {
				Log.d(EyeAIApp.APP_LOG_TAG, "[VoskModel] model unpacked")
				onModelLoaded()
			}
		}, { exception ->
			synchronized(this) {
				if (generation == modelLoadGeneration) modelLoading = false
			}
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"[VoskModel] Failed to unpack Vosk model '$modelName': $exception"
			)
		})
	}

	@Synchronized
	private fun ensureSoundPlayers() {
		if (activateSound == null) activateSound = AppSoundPlayer.create(context, R.raw.activate)
		if (deactivateSound == null) deactivateSound = AppSoundPlayer.create(context, R.raw.deactivate)
	}

	@Synchronized
	fun closeService() {
		++modelLoadGeneration
		modelLoading = false
		try {
			stopListening(playSound = false)
			speechService?.shutdown()
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] Exception closing service: ", e)
		} finally {
			speechService = null
			speechStreamService = null
			model = null
			isListening = false
			activateSound?.release()
			deactivateSound?.release()
			activateSound = null
			deactivateSound = null
		}
	}

	@Synchronized
	fun startListening(inputStream: InputStream? = null, playSound: Boolean = true) {
		if (isListening || speechService != null || speechStreamService != null) {
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"[VoskModel] startListening called while session active; stopping active session first."
			)
			stopListening(playSound = false)
		}
		if (model == null) {
			Log.w(EyeAIApp.APP_LOG_TAG, "[VoskModel] cannot startListening: model not loaded")
			return
		}
		try {
			val rec = Recognizer(model, SAMPLE_RATE)
			if (inputStream != null) {
				speechStreamService = SpeechStreamService(rec, inputStream, SAMPLE_RATE)
				speechStreamService?.start(recognitionListener)
			} else {
				speechService = SpeechService(rec, SAMPLE_RATE)
				speechService?.startListening(recognitionListener)
			}
			isListening = true
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"[VoskModel] started listening (stream=${inputStream != null})"
			)
			if (playSound) {
				activateSound?.seekTo(0)
				activateSound?.start()
			}
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] failed to create/start speechService: $e")
			isListening = false
		}
	}

	@Synchronized
	fun stopListening(playSound: Boolean = true) {
		if (!isListening) {
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"[VoskModel] stopListening called but not listening; ignoring."
			)
			return
		}
		try {
			speechService?.stop()
			speechStreamService?.stop()
			Log.d(EyeAIApp.APP_LOG_TAG, "[VoskModel] stopped listening")
			if (playSound) {
				deactivateSound?.seekTo(0)
				deactivateSound?.start()
			}
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[VoskModel] error while stopping speechService: ", e)
		} finally {
			speechService = null
			speechStreamService = null
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
