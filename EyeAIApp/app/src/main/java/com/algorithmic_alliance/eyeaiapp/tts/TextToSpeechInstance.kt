package com.algorithmic_alliance.eyeaiapp.tts

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

class TextToSpeechInstance (context: Context) : TextToSpeech.OnInitListener {

	public var tts: TextToSpeech? = null
	private var isInitialized = false

	init {
		// hier startest Du den Engine-Lifecycle
		tts = TextToSpeech(context, this)
	}




	override fun onInit(status: Int) {
		if (status == TextToSpeech.SUCCESS) {
			val result = tts?.setLanguage(Locale.GERMAN)
			if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
				Log.e("TTS", "Sprache nicht unterstützt oder Daten fehlen.")
			} else {
				isInitialized = true
				Log.d("TTS", "Initialisierung erfolgreich. Ready to speak!")
			}
		} else {
			Log.e("TTS", "Initialisierung fehlgeschlagen.")
		}
	}



	fun speak(text: String) {
		if (isInitialized) {
			tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "tts1")
		} else {
			Log.e("TTS", "TextToSpeech ist nicht initialisiert.")
		}


	}

	fun isSpeaking(): Boolean {
		return tts?.isSpeaking ?: false;
	}

	fun shutdown() {
		tts?.stop()
		tts?.shutdown()
	}
}