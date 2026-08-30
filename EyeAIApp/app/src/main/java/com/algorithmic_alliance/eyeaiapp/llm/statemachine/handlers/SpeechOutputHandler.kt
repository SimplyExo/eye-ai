package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import android.os.Handler
import android.os.Looper
import android.widget.TextView
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance

/** Sends local interaction responses through the existing UI and TTS path. */
class SpeechOutputHandler(
	private val textToSpeechInstance: TextToSpeechInstance,
	private val responseText: TextView?
) {
	private val mainHandler = Handler(Looper.getMainLooper())

	suspend fun speakAndHandleUi(text: String) {
		val toSpeak = text.trim()
		if (toSpeak.isEmpty()) {
			return
		}

		mainHandler.post { responseText?.text = toSpeak }
		textToSpeechInstance.speak(toSpeak)
	}
}
