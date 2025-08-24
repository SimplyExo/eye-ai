package com.algorithmic_alliance.eyeaiapp.tts

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import java.util.Locale
import kotlin.random.Random

class TextToSpeechInstance(context: Context, val onTTSFinishedSpeaking: () -> Unit) {
	var tts: TextToSpeech? = null
	private var isInitialized = false

	private var germanMaleVoice: Voice? = null
	private var germanFemaleVoice: Voice? = null


	private var onTTSInitListener = TextToSpeech.OnInitListener { status ->
		if (status == TextToSpeech.SUCCESS) {
			isInitialized = true

			val result = tts?.setLanguage(Locale.GERMAN)
			when (result) {
				TextToSpeech.SUCCESS -> Log.d("TTS", "Initialisierung erfolgreich. Ready to speak!")
				TextToSpeech.LANG_MISSING_DATA, TextToSpeech.LANG_NOT_SUPPORTED -> Log.e("TTS", "Sprache nicht unterstützt oder Daten fehlen.")
				else -> Log.e("TTS", "Konnte TTS Sprache nicht auf Deutsch stellen! Errorcode: $result")
			}
		} else {
			Log.e("TTS", "Initialisierung fehlgeschlagen. Fehler: $status")
		}
	}

	fun setVoice(number: Int){

		loadAvailableGermanVoices()

		if (germanFemaleVoice != null && number == 1) {
			tts?.voice = germanFemaleVoice
		}
		if (germanMaleVoice != null && number == 0) {
			tts?.voice = germanMaleVoice
		}

	}

	private fun loadAvailableGermanVoices() {
		try {
			// Find all German voices
			val germanVoices = tts?.voices?.filter { it.locale == Locale.GERMANY }

			// Find female and male voices


			if (germanVoices?.size!! > 1) {
				germanFemaleVoice = germanVoices[0]
				germanMaleVoice = germanVoices[1]
			} else {
				germanFemaleVoice = germanVoices.firstOrNull()
			}

			// Logging
			Log.d("TTS", "Gefundene weibliche Stimme: ${germanFemaleVoice?.name}")
			Log.d("TTS", "Gefundene männliche Stimme: ${germanMaleVoice?.name}")
			Log.d("TTS", tts?.voices.toString())

		} catch (e: Exception) {
			Log.e("TTS", "Fehler beim Laden der Stimmen: ", e)
		}
	}


	private var utteranceProgressListener = object : UtteranceProgressListener() {
		override fun onStart(utteranceId: String?) {
			Log.d(EyeAIApp.APP_LOG_TAG, "TTS onStart utteranceId=$utteranceId at ${System.currentTimeMillis()}")
		}

		override fun onDone(utteranceId: String?) {
			Log.d(EyeAIApp.APP_LOG_TAG, "TTS onDone utteranceId=$utteranceId at ${System.currentTimeMillis()}")
			onTTSFinishedSpeaking()

		}

		override fun onError(utteranceId: String?) {
			onTTSFinishedSpeaking()
		}

		override fun onStop(utteranceId: String?, interrupted: Boolean) {
			onTTSFinishedSpeaking()
		}
	}

	init {
		// Engine lifecycle
		tts = TextToSpeech(context, onTTSInitListener)
		tts?.setOnUtteranceProgressListener(utteranceProgressListener)
	}


	fun speak(text: String) {
		val utteranceId = "utt_${System.currentTimeMillis()}_${Random.nextInt(10000)}"

		if (isInitialized) {
			tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "tts1")
			Log.d(EyeAIApp.APP_LOG_TAG, "TextToSpeech.speak() called with utteranceId=$utteranceId")
		} else {
			Log.e("TTS", "TextToSpeech ist nicht initialisiert.")
		}
	}


	fun setSpeechRate(float: Float){
		tts?.setSpeechRate(float)
	}

	fun stop() {
		tts?.stop()
	}
	fun shutdown() {
		tts?.stop()
		tts?.shutdown()
	}
}