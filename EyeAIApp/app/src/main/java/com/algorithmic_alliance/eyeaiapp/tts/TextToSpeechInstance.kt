package com.algorithmic_alliance.eyeaiapp.tts

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

class TextToSpeechInstance(context: Context, val onTTSFinishedSpeaking: () -> Unit) {
	var tts: TextToSpeech? = null
	private var isInitialized = false

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

	private var utteranceProgressListener = object : UtteranceProgressListener() {
		override fun onStart(utteranceId: String?) {}

		override fun onDone(utteranceId: String?) {
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
		// hier startest Du den Engine-Lifecycle
		tts = TextToSpeech(context, onTTSInitListener)
		tts?.setOnUtteranceProgressListener(utteranceProgressListener)
	}


	fun speak(text: String) {
		if (isInitialized) {
			tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "tts1")
		} else {
			Log.e("TTS", "TextToSpeech ist nicht initialisiert.")
		}
	}

	fun stop() {
		tts?.stop()
	}
	fun shutdown() {
		tts?.stop()
		tts?.shutdown()
	}
}