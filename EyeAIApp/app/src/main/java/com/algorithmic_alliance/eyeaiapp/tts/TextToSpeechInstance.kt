package com.algorithmic_alliance.eyeaiapp.tts

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.audio.AppAudioOutput
import com.algorithmic_alliance.eyeaiapp.audio.PcmMixer
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.Flow
import java.util.Locale
import java.util.UUID
import kotlin.random.Random
import androidx.core.content.edit
import kotlin.time.Duration.Companion.milliseconds

class TextToSpeechInstance(
	private val context: Context, private val onTTSFinishedSpeaking: (() -> Unit)? = null
) : TextToSpeech.OnInitListener {

	var tts: TextToSpeech? = TextToSpeech(context, this)
	@Volatile private var isReady = false
	fun isReadyToSpeak(): Boolean = isReady && tts != null

	private val pendingCallbacks = java.util.concurrent.ConcurrentHashMap<String, () -> Unit>()
    private val startCallbacks = java.util.concurrent.ConcurrentHashMap<String, () -> Unit>()
    private val ownership=SpeechOwnership()
    private val sceneSpeech=SceneSpeechController(context)
    @Volatile private var externalOwner:String?=null
    private var externalCancelled:(()->Unit)?=null
    private var externalGeneration=0L
    private val deferredSpeech=java.util.ArrayDeque<()->Unit>()
    @Synchronized fun speakSpatialOwned(segments: List<com.algorithmic_alliance.eyeaiapp.scenenarrator.NarrationSegment>,owner:String,valid:()->Boolean,invalidated:Flow<Unit>,onStarted:()->Unit,onComplete:()->Unit,onCancelled:()->Unit):Boolean {
        if(!isReadyToSpeak()||isSpeaking()||segments.isEmpty())return false
        externalOwner=owner
        externalCancelled=onCancelled
        var started=false
        val notifyStarted={if(!started){started=true;onStarted()}}
        val generation=++externalGeneration
        fun release():Boolean {
            if(generation!=externalGeneration)return false
            externalOwner=null
            externalCancelled=null
            while(deferredSpeech.isNotEmpty())deferredSpeech.removeFirst().invoke()
            invokeOnFinishedWhenPlaybackStops()
            return true
        }
        sceneSpeech.start(segments,tts?.voice?.name,speechRate,valid,invalidated,
            notifyStarted,{if(release())onComplete()},
            {remaining->
                if(generation==externalGeneration) {
                    externalOwner=null
                    externalCancelled=null
                    speakOwned(remaining.joinToString(" "){it.text},owner,notifyStarted,onComplete)
                    while(deferredSpeech.isNotEmpty())deferredSpeech.removeFirst().invoke()
                }
            },{if(release())onCancelled()})
        return true
    }
    private fun cancelExternalSpeech() {
        if(externalOwner!=null) {externalGeneration++;externalOwner=null;sceneSpeech.stop();externalCancelled?.invoke();externalCancelled=null}
        deferredSpeech.clear()
    }
    fun stopIfOwnerPrefix(prefix: String): Boolean = synchronized(this) {
        if(externalOwner?.startsWith(prefix)!=true&&!ownership.canCancel(prefix))return@synchronized false
        stop()
        true
    }

	private val hasSpecificCallback = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()
	private class NetworkUtterance(
		val file: File,
		val destination: (String, ShortArray, () -> Unit) -> Unit,
	)
	private val networkUtterances = ConcurrentHashMap<String, NetworkUtterance>()
	private val networkDecodeDispatcher = Dispatchers.IO.limitedParallelism(1)

	// active utterance counter
	private var activeUtteranceCount = 0
	private val activeUtteranceIds = ConcurrentHashMap.newKeySet<String>()

	private val callbackScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

	// Backing field + public read-only property to avoid JVM setter name collision
	private var _speechRate: Float = 1.0f
	val speechRate: Float
		get() = _speechRate

	//needed to support streaming
	companion object {
		const val QUEUE_FLUSH = TextToSpeech.QUEUE_FLUSH
	}

	override fun onInit(status: Int) {
		if (status == TextToSpeech.SUCCESS) {
			try {
				tts?.language = Locale.GERMAN
			} catch (_: Exception) {
			}
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
				sharedPrefs.edit { remove("tts_voice") }
			}

		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Error loading TTS settings", e)
		}

	}

	@Synchronized
	private fun finishUtterance(utteranceId: String?) {
		if (utteranceId == null || !activeUtteranceIds.remove(utteranceId)) return
		decrementUtteranceCount()
		pendingCallbacks.remove(utteranceId)?.let { callback -> callbackScope.launch { callback() } }
		startCallbacks.remove(utteranceId)
		ownership.finished(utteranceId)
		hasSpecificCallback.remove(utteranceId)
		if (getActiveUtteranceCount() == 0) {
			invokeOnFinishedWhenPlaybackStops()
		}
	}


	// ------------------------------
	// Utterance listener
	// ------------------------------
    @Synchronized
    private fun notifyUtteranceStarted(id: String) {
        if (!activeUtteranceIds.contains(id)) return
        ownership.started(id)
        startCallbacks.remove(id)?.let { callback -> callbackScope.launch { callback() } }
    }

	private fun setupUtteranceListener() {
		tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
			override fun onStart(utteranceId: String?) {
				Log.d(EyeAIApp.APP_LOG_TAG, "TTS onStart utterance=$utteranceId")
                // File synthesis starts before network playback; notify after decoding instead.
                if (utteranceId != null && !networkUtterances.containsKey(utteranceId)) {
                    notifyUtteranceStarted(utteranceId)
                }
			}

			override fun onDone(utteranceId: String?) {
				val network = utteranceId?.let { networkUtterances[it] }
				if (network != null && utteranceId != null) {
					callbackScope.launch(networkDecodeDispatcher) {
						try {
							val samples = PcmMixer.decodeWave(network.file.readBytes())
							callbackScope.launch {
								if (networkUtterances[utteranceId] === network) {
									notifyUtteranceStarted(utteranceId)
									network.destination("tts", samples) {
										if (networkUtterances.remove(utteranceId, network)) finishUtterance(utteranceId)
									}
								}
							}
						} catch (error: Exception) {
							Log.e(EyeAIApp.APP_LOG_TAG, "Could not prepare network speech", error)
							callbackScope.launch {
								if (networkUtterances.remove(utteranceId, network)) finishUtterance(utteranceId)
							}
						} finally { network.file.delete() }
					}
					return
				}
				Log.d(EyeAIApp.APP_LOG_TAG, "TTS onDone utterance=$utteranceId")
				finishUtterance(utteranceId)

				if (getActiveUtteranceCount() == 0) {
					Log.d(
						EyeAIApp.APP_LOG_TAG,
						"No active utterances left -> schedule global callback when playback stops"
					)
					invokeOnFinishedWhenPlaybackStops()
				}
			}

			@Deprecated("Deprecated in Java")
			override fun onError(utteranceId: String?) {
				utteranceId?.let { networkUtterances.remove(it)?.file?.delete() }
				Log.e(EyeAIApp.APP_LOG_TAG, "TTS onError utterance=$utteranceId")
				finishUtterance(utteranceId)

				if (getActiveUtteranceCount() == 0) {
					Log.d(
						EyeAIApp.APP_LOG_TAG,
						"No active utterances left -> schedule global callback when playback stops"
					)
					invokeOnFinishedWhenPlaybackStops()
				}
			}

			override fun onStop(utteranceId: String?, interrupted: Boolean) {
				utteranceId?.let { networkUtterances.remove(it)?.file?.delete() }
				Log.d(
					EyeAIApp.APP_LOG_TAG,
					"TTS onStop utterance=$utteranceId interrupted=$interrupted"
				)
				finishUtterance(utteranceId)

				if (getActiveUtteranceCount() == 0) {
					Log.d(
						EyeAIApp.APP_LOG_TAG,
						"No active utterances left -> schedule global callback when playback stops"
					)
					invokeOnFinishedWhenPlaybackStops()
				}
			}
		})
	}


	//Simple stop method, supports stop button
	@Synchronized fun stop() {
		cancelNetworkSpeech()
		Log.w(EyeAIApp.APP_LOG_TAG, "TTS stop() called. Clearing queues and callbacks.")
		cancelExternalSpeech()
        if (!isReady) return
		try {
			tts?.stop()
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Error stopping TTS", e)
		}
		pendingCallbacks.clear()
        startCallbacks.clear()
ownership.clear()
		hasSpecificCallback.clear()
		activeUtteranceIds.clear()
		synchronized(this) { activeUtteranceCount = 0 }
		invokeOnFinishedWhenPlaybackStops()
	}

	//Should only be used if necessary, completely shuts down the TTS-Service
	fun shutdown() {
        cancelExternalSpeech()
        sceneSpeech.shutdown()
		cancelNetworkSpeech()
		pendingCallbacks.clear()
        startCallbacks.clear()
ownership.clear()
		hasSpecificCallback.clear()
		activeUtteranceIds.clear()
		synchronized(this) { activeUtteranceCount = 0 }
		try {
			tts?.stop()
		} catch (_: Exception) {
		}
		try {
			tts?.shutdown()
		} catch (_: Exception) {
		}
		tts = null
		callbackScope.coroutineContext.cancel()
	}

	//ensuring silence when starting to listen with vosk
	suspend fun awaitSilence(
		quietMs: Long = 500L, maxWaitMs: Long = 15_000L, pollIntervalMs: Long = 50L
	): Boolean {
		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"awaitSilence: Starting wait for ${quietMs}ms of silence (max ${maxWaitMs}ms)."
		)
		val start = System.currentTimeMillis()
		var silentStart = -1L
		while (System.currentTimeMillis() - start < maxWaitMs) {
			val isSpeakingFlag = tts?.isSpeaking == true
			val utteranceCount = getActiveUtteranceCount()


			val speakingNow = isSpeakingFlag || (utteranceCount > 0) || externalOwner!=null

			if (speakingNow) {
				if (silentStart != -1L || (System.currentTimeMillis() - start) % 500 < pollIntervalMs) {
					Log.d(
						EyeAIApp.APP_LOG_TAG,
						"awaitSilence: Not silent. isSpeaking=$isSpeakingFlag, activeUtterances=$utteranceCount"
					)
				}
				// Reset silent window
				silentStart = -1L
			} else {
				if (silentStart == -1L) {
					silentStart = System.currentTimeMillis()
					Log.d(EyeAIApp.APP_LOG_TAG, "awaitSilence: Potential silent period started.")
				} else {
					if (System.currentTimeMillis() - silentStart >= quietMs) {
						Log.d(
							EyeAIApp.APP_LOG_TAG,
							"awaitSilence: stable quiet window ${quietMs}ms reached"
						)
						return true
					}
				}
			}
			delay(pollIntervalMs.milliseconds)
		}
		Log.w(
			EyeAIApp.APP_LOG_TAG,
			"awaitSilence: timeout after ${maxWaitMs}ms (silentStart=$silentStart)"
		)
		return false
	}

	//ensures that there is a slight, stable delay after speaking and listening
	private var invokeFinishedJob: Job? = null

	private fun invokeOnFinishedWhenPlaybackStops() {
		val callback = onTTSFinishedSpeaking ?: return

		// cancel old wait and restart if there is an active one
		invokeFinishedJob?.cancel()

		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"invokeOnFinishedWhenPlaybackStops: launching wait-for-silence job."
		)
		invokeFinishedJob = callbackScope.launch {
			try {
				val silent = try {
					awaitSilence(quietMs = 500L, maxWaitMs = 15_000L)
				} catch (e: Exception) {
					Log.e(EyeAIApp.APP_LOG_TAG, "Exception in awaitSilence", e)
					false
				}

				Log.d(
					EyeAIApp.APP_LOG_TAG,
					"invokeOnFinishedWhenPlaybackStops: silent=$silent -> invoking global callback"
				)
				try {
					callback()
				} catch (e: Exception) {
					Log.e(EyeAIApp.APP_LOG_TAG, "Exception in onTTSFinishedSpeaking", e)
				}
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
	fun speak(text: String, queueMode: Int = QUEUE_FLUSH, onComplete: (() -> Unit)? = null) = enqueue(text,queueMode,null,null,onComplete)

    @Synchronized fun speakOwned(text: String,owner: String,onStarted: (() -> Unit)?=null,onComplete: (() -> Unit)?=null): Boolean {
        // Atomic admission with all other speak calls. Never flush a dialogue
        // that raced with the runtime's readiness check.
        if(!isReadyToSpeak() || isSpeaking())return false
        enqueue(text,TextToSpeech.QUEUE_ADD,owner,onStarted,onComplete)
        return true
    }

    @Synchronized private fun enqueue(text: String,queueMode: Int,owner: String?,onStarted: (() -> Unit)?,onComplete: (() -> Unit)?) {
        if (AppAudioOutput.isNetworkOutput && queueMode == QUEUE_FLUSH) stop()
		if (!isReady) {
			if (onComplete != null) {
				onComplete.invoke()
			} else {
				Log.e(EyeAIApp.APP_LOG_TAG, "TTS not ready, cannot speak.")
			}
			return
		}

        if(externalOwner!=null) {
            if(queueMode==QUEUE_FLUSH)cancelExternalSpeech()
            else {deferredSpeech.addLast {enqueue(text,queueMode,owner,onStarted,onComplete)};return}
        }
        if (queueMode == QUEUE_FLUSH) {
            pendingCallbacks.clear()
            startCallbacks.clear()
            hasSpecificCallback.clear()
            activeUtteranceIds.clear()
            ownership.clear()
            activeUtteranceCount = 0
        }
		if (onComplete != null) {
			val utteranceId = UUID.randomUUID().toString()
            ownership.enqueue(utteranceId,owner ?: "dialog",queueMode==QUEUE_FLUSH)
            onStarted?.let {startCallbacks[utteranceId]=it}
			pendingCallbacks[utteranceId] = onComplete
			hasSpecificCallback.add(utteranceId)
			incrementUtteranceCount(utteranceId)
			val params = Bundle().apply {
				putString(
					TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId
				)
			}
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"speak enqueued utterance=$utteranceId (with specific callback) and queueMode=$queueMode"
			)
			speakWithOutput(text, queueMode, params, utteranceId)
		} else {
			val utteranceId = "utt_${System.currentTimeMillis()}_${Random.nextInt(10000)}"
            ownership.enqueue(utteranceId,owner ?: "dialog",queueMode==QUEUE_FLUSH)
            onStarted?.let {startCallbacks[utteranceId]=it}
			incrementUtteranceCount(utteranceId)
			val queueModeStr = if (queueMode == QUEUE_FLUSH) "QUEUE_FLUSH" else "QUEUE_ADD"
			val params = Bundle().apply {
				putString(
					TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId
				)
			}
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"speak enqueued utterance=$utteranceId with queueMode=$queueModeStr"
			)
			speakWithOutput(text, queueMode, params, utteranceId)
		}
	}

	private fun speakWithOutput(text: String, queueMode: Int, params: Bundle, utteranceId: String) {
		val destination = AppAudioOutput.destination()
		if (destination == null) {
			if (tts?.speak(text, queueMode, params, utteranceId) != TextToSpeech.SUCCESS) {
				finishUtterance(utteranceId)
			}
			return
		}
		try {
			val file = File.createTempFile("network-tts-", ".wav", context.cacheDir)
			networkUtterances[utteranceId] = NetworkUtterance(file, destination)
			if (tts?.synthesizeToFile(text, params, file, utteranceId) != TextToSpeech.SUCCESS) {
				networkUtterances.remove(utteranceId)?.file?.delete()
				finishUtterance(utteranceId)
			}
		} catch (error: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Could not synthesize network speech", error)
			networkUtterances.remove(utteranceId)?.file?.delete()
			finishUtterance(utteranceId)
		}
	}

	private fun cancelNetworkSpeech() {
		AppAudioOutput.remove("tts")
		val cancelled = networkUtterances.values.toList()
		networkUtterances.clear()
		cancelled.forEach { it.file.delete() }
	}


	// ------------------------------
	// Settings
	// ------------------------------

	/**
	 * Public setter method (keeps API stable for callers that expect setSpeechRate(rate: Float))
	 * Uses `_speechRate` to avoid any JVM setter collision.
	 */
	fun setSpeechRate(rate: Float) {
		if (isReady) {
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
			"[TTS][VOICES] engine=${runCatching { engine.defaultEngine }.getOrNull()} " + "current=${currentVoice?.name} default=${defaultVoice?.name} " + "available=${
				availableVoices.joinToString { voice ->
					"${voice.name}[${voice.locale},features=${voice.features}]"
				}
			}")

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
		val selectedVoice =
			availableVoices.firstOrNull { it.name == selectedDescriptor.name } ?: run {
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
		val availableDescriptors =
			availableVoices.filter { it.name != failedVoiceName }.map { it.toTtsVoiceDescriptor() }
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
	private fun incrementUtteranceCount(utteranceId: String) {
		activeUtteranceIds.add(utteranceId)
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

	fun isSpeaking(): Boolean = externalOwner!=null || tts?.isSpeaking == true || getActiveUtteranceCount() > 0
}
