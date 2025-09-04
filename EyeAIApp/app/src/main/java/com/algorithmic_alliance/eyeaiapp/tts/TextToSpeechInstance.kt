package com.algorithmic_alliance.eyeaiapp.tts

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.Locale
import java.util.UUID
import kotlin.random.Random

class TextToSpeechInstance(
	context: Context,
	private val onTTSFinishedSpeaking: (() -> Unit)? = null
) : TextToSpeech.OnInitListener {

	private var tts: TextToSpeech? = TextToSpeech(context, this)
	private var isReady = false

	private val pendingCallbacks = mutableMapOf<String, () -> Unit>()
	private val hasSpecificCallback = mutableSetOf<String>()

	// active utterance counter
	private var activeUtteranceCount = 0

	private var germanMaleVoice: Voice? = null
	private var germanFemaleVoice: Voice? = null

	private val callbackScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

	// Backing field + public read-only property to avoid JVM setter name collision
	private var _speechRate: Float = 1.0f
	val speechRate: Float
		get() = _speechRate

	//needed to support streaming
	companion object {
		const val QUEUE_FLUSH = TextToSpeech.QUEUE_FLUSH
		const val QUEUE_ADD = TextToSpeech.QUEUE_ADD
	}

	override fun onInit(status: Int) {
		if (status == TextToSpeech.SUCCESS) {
			try { tts?.language = Locale.GERMAN } catch (_: Exception) {}
			isReady = true
			Log.d(EyeAIApp.APP_LOG_TAG, "TextToSpeech initialized.")
			setupUtteranceListener()
		} else {
			isReady = false
			Log.e(EyeAIApp.APP_LOG_TAG, "Failed to init TextToSpeech.")
		}
	}



	// ------------------------------
	// Utterance listener
	// ------------------------------
	private fun setupUtteranceListener() {
		tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
			override fun onStart(utteranceId: String?) {
				Log.d(EyeAIApp.APP_LOG_TAG, "TTS onStart utterance=$utteranceId")
				// Markiere, dass Wiedergabe aktiv ist (so lange mindestens eine Utterance läuft oder gerade lief)
			}

			override fun onDone(utteranceId: String?) {
				Log.d(EyeAIApp.APP_LOG_TAG, "TTS onDone utterance=$utteranceId")
				utteranceId?.let { id ->
					// Zähler zuerst reduzieren (wichtig)
					decrementUtteranceCount()

					// jetzt spezifischen Callback ausführen (falls vorhanden)
					pendingCallbacks.remove(id)?.invoke()
					hasSpecificCallback.remove(id)

					// wenn keine Utterances mehr aktiv sind -> prüfen wir Playback und rufen globales Callback
					if (getActiveUtteranceCount() == 0) {
						Log.d(EyeAIApp.APP_LOG_TAG, "No active utterances left -> schedule global callback when playback stops")
						invokeOnFinishedWhenPlaybackStops()
					}
				}
			}

			override fun onError(utteranceId: String?) {
				Log.e(EyeAIApp.APP_LOG_TAG, "TTS onError utterance=$utteranceId")
				utteranceId?.let { id ->
					decrementUtteranceCount()
					pendingCallbacks.remove(id)?.invoke()
					hasSpecificCallback.remove(id)
					if (getActiveUtteranceCount() == 0) {
						invokeOnFinishedWhenPlaybackStops()
					}
				}
			}

			override fun onStop(utteranceId: String?, interrupted: Boolean) {
				Log.d(EyeAIApp.APP_LOG_TAG, "TTS onStop utterance=$utteranceId interrupted=$interrupted")
				utteranceId?.let { id ->
					decrementUtteranceCount()
					pendingCallbacks.remove(id)?.invoke()
					hasSpecificCallback.remove(id)
					if (getActiveUtteranceCount() == 0) {
						invokeOnFinishedWhenPlaybackStops()
					}
				}
			}
		})
	}


	//Simple stop method, supports stop button
	fun stop() {
		Log.w(EyeAIApp.APP_LOG_TAG, "TTS stop() called. Clearing queues and callbacks.")
		if (!isReady) return
		try {
			tts?.stop()
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Error stopping TTS", e)
		}
		pendingCallbacks.clear()
		hasSpecificCallback.clear()
		synchronized(this) { activeUtteranceCount = 0 }
		invokeOnFinishedWhenPlaybackStops()
	}
	//Should only be used if necessary, completely shuts down the TTS-Service
	fun shutdown() {
		pendingCallbacks.clear()
		hasSpecificCallback.clear()
		synchronized(this) { activeUtteranceCount = 0 }
		try { tts?.stop() } catch (_: Exception) {}
		try { tts?.shutdown() } catch (_: Exception) {}
		tts = null
		callbackScope.coroutineContext.cancel()
	}

	//ensuring silence when starting to listen with vosk
	suspend fun awaitSilence(quietMs: Long = 500L, maxWaitMs: Long = 15_000L, pollIntervalMs: Long = 50L): Boolean {
		Log.d(EyeAIApp.APP_LOG_TAG, "awaitSilence: Starting wait for ${quietMs}ms of silence (max ${maxWaitMs}ms).")
		val start = System.currentTimeMillis()
		var silentStart = -1L
		while (System.currentTimeMillis() - start < maxWaitMs) {
			val isSpeakingFlag = tts?.isSpeaking == true
			val utteranceCount = getActiveUtteranceCount()


			val speakingNow = isSpeakingFlag || (utteranceCount > 0)

			if (speakingNow) {
				if (silentStart != -1L || (System.currentTimeMillis() - start) % 500 < pollIntervalMs) {
					Log.d(EyeAIApp.APP_LOG_TAG, "awaitSilence: Not silent. isSpeaking=$isSpeakingFlag, activeUtterances=$utteranceCount")
				}
				// Reset silent window
				silentStart = -1L
			} else {
				if (silentStart == -1L) {
					silentStart = System.currentTimeMillis()
					Log.d(EyeAIApp.APP_LOG_TAG, "awaitSilence: Potential silent period started.")
				} else {
					if (System.currentTimeMillis() - silentStart >= quietMs) {
						Log.d(EyeAIApp.APP_LOG_TAG, "awaitSilence: stable quiet window ${quietMs}ms reached")
						return true
					}
				}
			}
			delay(pollIntervalMs)
		}
		Log.w(EyeAIApp.APP_LOG_TAG, "awaitSilence: timeout after ${maxWaitMs}ms (silentStart=$silentStart)")
		return false
	}

	//ensures that there is a slight, stable delay after speaking and listening
	private var invokeFinishedJob: Job? = null

	private fun invokeOnFinishedWhenPlaybackStops() {
		val callback = onTTSFinishedSpeaking ?: return

		// cancel old wait and restart if there is an active one
		invokeFinishedJob?.cancel()

		Log.d(EyeAIApp.APP_LOG_TAG, "invokeOnFinishedWhenPlaybackStops: launching wait-for-silence job.")
		invokeFinishedJob = callbackScope.launch {
			try {
				val silent = try {
					awaitSilence(quietMs = 500L, maxWaitMs = 15_000L)
				} catch (e: Exception) {
					Log.e(EyeAIApp.APP_LOG_TAG, "Exception in awaitSilence", e)
					false
				}

				Log.d(EyeAIApp.APP_LOG_TAG, "invokeOnFinishedWhenPlaybackStops: silent=$silent -> invoking global callback")
				try { callback() } catch (e: Exception) { Log.e(EyeAIApp.APP_LOG_TAG, "Exception in onTTSFinishedSpeaking", e) }
			} finally {
				invokeFinishedJob = null
			}
		}
	}


	// ------------------------------
	// Speak API
	// ------------------------------

	/**
	 * Unified speak function with defaults, replaced older variant with multiple overloads in order to avoid confusion
	 * - queueMode defaults to QUEUE_FLUSH so calls like speak(text) { ... } can be used for responses that don't need to support streaming responses
	 * - onComplete if provided registers a specific callback and uses a specific utteranceId and params
	 */
	@JvmOverloads
	fun speak(text: String, queueMode: Int = QUEUE_FLUSH, onComplete: (() -> Unit)? = null) {
		if (!isReady) {
			if (onComplete != null) {
				onComplete.invoke()
			} else {
				Log.e(EyeAIApp.APP_LOG_TAG, "TTS not ready, cannot speak.")
			}
			return
		}

		if (onComplete != null) {
			val utteranceId = UUID.randomUUID().toString()
			pendingCallbacks[utteranceId] = onComplete
			hasSpecificCallback.add(utteranceId)
			incrementUtteranceCount()
			val params = Bundle().apply { putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId) }
			Log.d(EyeAIApp.APP_LOG_TAG, "speak enqueued utterance=$utteranceId (with specific callback) and queueMode=$queueMode")
			tts?.speak(text, queueMode, params, utteranceId)
		} else {
			val utteranceId = "utt_${System.currentTimeMillis()}_${Random.nextInt(10000)}"
			incrementUtteranceCount()
			val queueModeStr = if (queueMode == QUEUE_FLUSH) "QUEUE_FLUSH" else "QUEUE_ADD"
			Log.d(EyeAIApp.APP_LOG_TAG, "speak enqueued utterance=$utteranceId with queueMode=$queueModeStr")
			tts?.speak(text, queueMode, null, utteranceId)
		}
	}

	// ------------------------------
	// Settings
	// ------------------------------

	/**
	 * Public setter method (keeps API stable for callers that expect setSpeechRate(rate: Float))
	 * Uses `_speechRate` to avoid any JVM setter collision.
	 */
	fun setSpeechRate(rate: Float) {
		if (isReady){
			tts?.setSpeechRate(rate)
			_speechRate = rate
		} else {
			Log.e(EyeAIApp.APP_LOG_TAG, "TTS not ready, cannot set speech rate")
		}
	}

	//Method used to switch between male and female voices, numbers are generated by the LLM and used to avoid any useless responses

	fun setVoice(number: Int) {
		loadAvailableGermanVoices()
		when (number) {
			0 -> if (germanMaleVoice != null) tts?.voice = germanMaleVoice
			1 -> if (germanFemaleVoice != null) tts?.voice = germanFemaleVoice
			else -> Log.w(EyeAIApp.APP_LOG_TAG, "Invalid voice number: $number")
		}
	}

	private fun loadAvailableGermanVoices() {
		try {
			val germanVoices = tts?.voices?.filter { it.locale == Locale.GERMANY || it.locale == Locale.GERMAN }
			if (!germanVoices.isNullOrEmpty()) {
				germanFemaleVoice = germanVoices.getOrNull(0)
				germanMaleVoice = germanVoices.getOrNull(1)
			}
		} catch (e: Exception) { Log.e(EyeAIApp.APP_LOG_TAG, "Error loading voices", e) }
	}

	// ------------------------------
	// counter helpers, needed for debugging only
	// ------------------------------
	@Synchronized
	private fun incrementUtteranceCount() {
		activeUtteranceCount++
		Log.d(EyeAIApp.APP_LOG_TAG, "TTS activeUtterances increment -> $activeUtteranceCount")
	}

	@Synchronized
	private fun decrementUtteranceCount() {
		if (activeUtteranceCount > 0) activeUtteranceCount--
		Log.d(EyeAIApp.APP_LOG_TAG, "TTS activeUtterances decrement -> $activeUtteranceCount")
	}

	@Synchronized
	fun getActiveUtteranceCount(): Int = activeUtteranceCount

}
