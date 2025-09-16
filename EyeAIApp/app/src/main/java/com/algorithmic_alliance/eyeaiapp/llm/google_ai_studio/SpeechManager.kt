package com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio

import android.annotation.SuppressLint
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.LLMStreamingHandler
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance

object SpeechManager {
	lateinit var tts: TextToSpeechInstance

	@SuppressLint("StaticFieldLeak")
	var stream: LLMStreamingHandler? = null
	var llm: GoogleAIStudioLLM? = null


	fun forceStop() {


		llm?.stopCurrentStream()

		stream?.let {
			if (it.isStreaming()) {
				it.stopStreaming()
			}
		}

		if (tts.isSpeaking()) {
			tts.stop()
		}
	}
}
