package com.algorithmic_alliance.eyeaiapp.tts

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

class TextToSpeechInstance () : TextToSpeech.OnInitListener {

	public var tts: TextToSpeech? = null
	private var isInitialized = false



	override fun onInit(status: Int) {
		if (status == TextToSpeech.SUCCESS) {
			val result = tts?.setLanguage(Locale.GERMANY)
			if(result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED){
				Log.e("TTS", "Sprache nicht unterstützt")
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