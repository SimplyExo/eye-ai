package com.algorithmic_alliance.eyeaiapp.tts

import android.content.Context
import android.media.AudioAttributes
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
	private val context: Context,
	private val onTTSFinishedSpeaking: (() -> Unit)? = null
) : TextToSpeech.OnInitListener {

	var tts: TextToSpeech? = TextToSpeech(context, this)
	private var isReady = false

	private val pendingCallbacks = mutableMapOf<String, () -> Unit>()
	private val hasSpecificCallback = mutableSetOf<String>()

	// active utterance counter
	private var activeUtteranceCount = 0

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
			Log.d(EyeAIApp.APP_LOG_TAG, "Loading saved settings...")
			loadSavedTTSSettings()
			Log.d(EyeAIApp.APP_LOG_TAG, "Saved settings loaded!")
			setupUtteranceListener()
		} else {
			isReady = false
			Log.e(EyeAIApp.APP_LOG_TAG, "Failed to init TextToSpeech.")
		}
	}

	private fun loadSavedTTSSettings() {
		try {
			val sharedPrefs = context.getSharedPreferences("tts_settings", Context.MODE_PRIVATE)

			// Load tts speed
			val savedSpeechRate = sharedPrefs.getFloat("tts_speech_rate", 1.0f)
			if (savedSpeechRate != 1.0f) {
				setSpeechRate(savedSpeechRate)
			}

			// Load voice
			val savedVoice = sharedPrefs.getInt("tts_voice", -1)
			if (savedVoice != -1 && !setVoice(savedVoice)) {
				Log.w(
					EyeAIApp.APP_LOG_TAG,
					"Saved TTS speaker is not available on this device; keeping the active voice."
				)
				sharedPrefs.edit().remove("tts_voice").apply()
			}

		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Error loading TTS settings", e)
		}

	}

	private fun finishUtterance(utteranceId: String?){
		decrementUtteranceCount()
		utteranceId?.let {
			id ->
			pendingCallbacks.remove(id)?.invoke()
			hasSpecificCallback.remove(id)
		}
		if(getActiveUtteranceCount() == 0){
			invokeOnFinishedWhenPlaybackStops()
		}
	}


	// ------------------------------
	// Utterance listener
	// ------------------------------
	private fun setupUtteranceListener() {
		tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
			override fun onStart(utteranceId: String?) {
				Log.d(EyeAIApp.APP_LOG_TAG, "TTS onStart utterance=$utteranceId")
			}

			override fun onDone(utteranceId: String?) {
				Log.d(EyeAIApp.APP_LOG_TAG, "TTS onDone utterance=$utteranceId")
				finishUtterance(utteranceId)

				if (getActiveUtteranceCount() == 0) {
					Log.d(EyeAIApp.APP_LOG_TAG, "No active utterances left -> schedule global callback when playback stops")
					invokeOnFinishedWhenPlaybackStops()
				}
			}

			override fun onError(utteranceId: String?) {
				Log.e(EyeAIApp.APP_LOG_TAG, "TTS onError utterance=$utteranceId")
				finishUtterance(utteranceId)

				if (getActiveUtteranceCount() == 0) {
					Log.d(EyeAIApp.APP_LOG_TAG, "No active utterances left -> schedule global callback when playback stops")
					invokeOnFinishedWhenPlaybackStops()
				}
			}

			override fun onStop(utteranceId: String?, interrupted: Boolean) {
				Log.d(EyeAIApp.APP_LOG_TAG, "TTS onStop utterance=$utteranceId interrupted=$interrupted")
				finishUtterance(utteranceId)

				if (getActiveUtteranceCount() == 0) {
					Log.d(EyeAIApp.APP_LOG_TAG, "No active utterances left -> schedule global callback when playback stops")
					invokeOnFinishedWhenPlaybackStops()
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
			val params = Bundle().apply { putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId) }
			Log.d(EyeAIApp.APP_LOG_TAG, "speak enqueued utterance=$utteranceId with queueMode=$queueModeStr")
			tts?.speak(text, queueMode, params, utteranceId)
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

	/**
	 * Selects a speaker only from the voices exposed by the active Android TTS
	 * engine. The integer values are the existing settings wire format:
	 * 0=female and 1=male.
	 *
	 * The return value is deliberately false when a requested speaker could not
	 * be applied, including when only a safe fallback could be kept active. That
	 * lets the settings layer avoid persisting or announcing an unapplied change.
	 */
	fun setVoice(number: Int): Boolean {
		if (!isReady) {
			Log.w(EyeAIApp.APP_LOG_TAG, "TTS not ready, cannot set speaker=$number")
			return false
		}
		if (number !in setOf(TtsVoiceSelector.FEMALE, TtsVoiceSelector.MALE)) {
			Log.w(EyeAIApp.APP_LOG_TAG, "Invalid speaker number: $number")
			return false
		}

		val engine = tts ?: run {
			Log.w(EyeAIApp.APP_LOG_TAG, "TTS engine is unavailable, cannot set speaker=$number")
			return false
		}
		val availableVoices = try {
			engine.voices?.toList().orEmpty()
		} catch (error: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Could not read voices from the active TTS engine", error)
			return false
		}
		val currentVoice = runCatching { engine.voice }.getOrNull()
		val defaultVoice = runCatching { engine.defaultVoice }.getOrNull()
		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"[TTS][VOICES] engine=${runCatching { engine.defaultEngine }.getOrNull()} " +
				"current=${currentVoice?.name} default=${defaultVoice?.name} " +
				"available=${availableVoices.joinToString { voice ->
					"${voice.name}[${voice.locale},features=${voice.features}]"
				}}"
		)

		val descriptors = availableVoices.map { it.toTtsVoiceDescriptor() }
		val selection = TtsVoiceSelector.select(
			requestedSpeaker = number,
			voices = descriptors,
			currentVoiceName = currentVoice?.name,
			defaultVoiceName = defaultVoice?.name
		)
		val selectedDescriptor = selection.voice ?: run {
			Log.w(EyeAIApp.APP_LOG_TAG, "No usable TTS voice is available for speaker=$number")
			return false
		}
		val selectedVoice = availableVoices.firstOrNull { it.name == selectedDescriptor.name } ?: run {
			Log.w(
				EyeAIApp.APP_LOG_TAG,
				"Selected TTS voice disappeared from the active engine catalog: ${selectedDescriptor.name}"
			)
			return false
		}

		if (applyVoice(engine, selectedVoice)) {
			if (selection.requestedSpeakerAvailable) {
				Log.i(
					EyeAIApp.APP_LOG_TAG,
					"[TTS][VOICE] speaker=$number applied=${selectedVoice.name}"
				)
				return true
			}
			Log.w(
				EyeAIApp.APP_LOG_TAG,
				"[TTS][VOICE] speaker=$number unavailable; kept safe voice=${selectedVoice.name}"
			)
			return false
		}

		Log.w(
			EyeAIApp.APP_LOG_TAG,
			"[TTS][VOICE] setVoice failed for requested=${selectedVoice.name}; restoring a safe voice"
		)
		restoreSafeVoice(
			engine = engine,
			availableVoices = availableVoices,
			previousVoice = currentVoice,
			defaultVoice = defaultVoice,
			failedVoiceName = selectedVoice.name
		)
		return false
	}

	private fun applyVoice(engine: TextToSpeech, voice: Voice): Boolean {
		val result = try {
			engine.setVoice(voice)
		} catch (error: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Exception while setting TTS voice ${voice.name}", error)
			return false
		}
		val effectiveVoiceName = runCatching { engine.voice?.name }.getOrNull()
		val applied = result == TextToSpeech.SUCCESS && effectiveVoiceName == voice.name
		if (!applied) {
			Log.w(
				EyeAIApp.APP_LOG_TAG,
				"[TTS][VOICE] rejected=${voice.name} result=$result effective=$effectiveVoiceName"
			)
		}
		return applied
	}

	private fun restoreSafeVoice(
		engine: TextToSpeech,
		availableVoices: List<Voice>,
		previousVoice: Voice?,
		defaultVoice: Voice?,
		failedVoiceName: String
	) {
		val availableDescriptors = availableVoices
			.filter { it.name != failedVoiceName }
			.map { it.toTtsVoiceDescriptor() }
		val safeDescriptor = TtsVoiceSelector.select(
			requestedSpeaker = -1,
			voices = availableDescriptors,
			currentVoiceName = previousVoice?.name,
			defaultVoiceName = defaultVoice?.name
		).voice ?: return
		val safeVoice = availableVoices.firstOrNull { it.name == safeDescriptor.name } ?: return
		if (applyVoice(engine, safeVoice)) {
			Log.i(EyeAIApp.APP_LOG_TAG, "[TTS][VOICE] safe fallback active=${safeVoice.name}")
		} else {
			Log.e(EyeAIApp.APP_LOG_TAG, "[TTS][VOICE] safe fallback could not be applied")
		}
	}

	private fun Voice.toTtsVoiceDescriptor(): TtsVoiceDescriptor = TtsVoiceDescriptor(
		name = name,
		locale = locale,
		features = features.orEmpty().toSet(),
		isNetworkConnectionRequired = isNetworkConnectionRequired,
		quality = quality,
		latency = latency
	)

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

	fun isSpeaking(): Boolean = tts?.isSpeaking == true || getActiveUtteranceCount() > 0
}
