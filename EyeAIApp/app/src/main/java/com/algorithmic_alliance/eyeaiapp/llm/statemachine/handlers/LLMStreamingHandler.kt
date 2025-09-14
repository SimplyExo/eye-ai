package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers


import android.util.Log
import android.widget.TextView
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio.GoogleAIStudioLLM
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.lang.StringBuilder

class LLMStreamingHandler(
	private val textToSpeechInstance: TextToSpeechInstance,
	private val llmResponseText: TextView?,
	private val eyeAIApp: EyeAIApp,
	private val onStreamingComplete: () -> Unit
) {
	private val sentenceBuffer = StringBuilder()
	private var isFirstStreamChunk = true
	private var lastEmittedChunk: String? = null
	private val sentenceDelimiters = charArrayOf('.', '!', '?')

	@Volatile
	private var isCurrentlyStreaming = false

	fun isStreaming(): Boolean = isCurrentlyStreaming

	suspend fun generateAndStreamResponse(llm: GoogleAIStudioLLM, prompt: String) {
		Log.d(EyeAIApp.APP_LOG_TAG, "Starting stream with prompt: '${prompt.take(100)}...'")
		isCurrentlyStreaming = true
		synchronized(sentenceBuffer) { sentenceBuffer.clear() }
		isFirstStreamChunk = true
		withContext(Dispatchers.Main) { llmResponseText?.text = "" }

		try {
			llm.generateStream(
				command = prompt,
				onChunk = { chunk -> handleStreamChunk(chunk) },
				onComplete = { handleStreamComplete() },
				onError = { e -> handleStreamError(e) }
			)
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Exception starting stream", e)
			speakAndHandleUi("Entschuldigung, beim Starten der Anfrage ist ein Fehler aufgetreten.")
			isCurrentlyStreaming = false
			onStreamingComplete()
		}
	}

	private fun handleStreamChunk(chunk: String) {
		try {
			Log.v(EyeAIApp.APP_LOG_TAG, "Stream chunk received: '$chunk'")
			val normalized = chunk.replace("\r", " ").replace("\n", " ").replace(Regex("\\s+"), " ").trim()
			if (normalized.isEmpty() || lastEmittedChunk == normalized) return

			synchronized(sentenceBuffer) {
				if (sentenceBuffer.isNotEmpty()) {
					val lastChar = sentenceBuffer.last()
					val firstChar = normalized.first()
					val punctuation = setOf('.', '!', '?', ',', ';', ':', '"', '\'', ')', '(')
					if (!lastChar.isWhitespace() && !punctuation.contains(firstChar)) {
						sentenceBuffer.append(' ')
					}
				}
				sentenceBuffer.append(normalized)
				Log.v(EyeAIApp.APP_LOG_TAG, "Sentence buffer is now: '$sentenceBuffer'")
			}

			lastEmittedChunk = normalized
			processSentenceBuffer()
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Exception in chunk handler", e)
		}
	}

	private fun handleStreamComplete() {
		Log.d(EyeAIApp.APP_LOG_TAG, "Stream completed from network.")
		synchronized(sentenceBuffer) {
			if (sentenceBuffer.isNotEmpty()) {
				val remainingText = sentenceBuffer.toString().trim()
				Log.d(EyeAIApp.APP_LOG_TAG, "Stream complete. Speaking remaining text: '$remainingText'")
				if (remainingText.isNotEmpty()) {
					speakAndDisplaySentence(remainingText, isLastChunk = true)
				}
				sentenceBuffer.clear()
			} else {
				Log.d(EyeAIApp.APP_LOG_TAG, "Stream complete, buffer empty. Invoking completion callback.")
				isCurrentlyStreaming = false
				onStreamingComplete()
			}
		}
		lastEmittedChunk = null
	}

	private fun handleStreamError(e: Exception) {
		Log.e(EyeAIApp.APP_LOG_TAG, "LLM stream error", e)
		isCurrentlyStreaming = false
		CoroutineScope(Dispatchers.Main).launch {
			speakAndHandleUi("Entschuldigung, bei der Anfrage ist ein Fehler aufgetreten.")
		}
	}

	private fun processSentenceBuffer() {
		while (true) {
			val nextDelimiterIndex = sentenceBuffer.indexOfAny(sentenceDelimiters)
			if (nextDelimiterIndex == -1) break

			val sentence = sentenceBuffer.substring(0, nextDelimiterIndex + 1)
			Log.d(EyeAIApp.APP_LOG_TAG, "Extracted sentence to speak: '$sentence'")
			speakAndDisplaySentence(sentence.trim(), isLastChunk = false)
			sentenceBuffer.delete(0, nextDelimiterIndex + 1)
		}
	}

	private fun speakAndDisplaySentence(sentence: String, isLastChunk: Boolean = false) {
		if (sentence.isBlank()) {
			if (isLastChunk) {
				isCurrentlyStreaming = false
				onStreamingComplete()
			}
			return
		}

		CoroutineScope(Dispatchers.Main).launch { llmResponseText?.append("$sentence ") }

		val queueMode = if (isFirstStreamChunk) {
			isFirstStreamChunk = false
			TextToSpeechInstance.QUEUE_FLUSH
		} else {
			TextToSpeechInstance.QUEUE_ADD
		}

		val queueModeStr = if(queueMode == TextToSpeechInstance.QUEUE_FLUSH) "FLUSH" else "ADD"
		Log.d(EyeAIApp.APP_LOG_TAG, "speakAndDisplaySentence: isLastChunk=$isLastChunk, queueMode=$queueModeStr, sentence='$sentence'")

		if (isLastChunk) {
			textToSpeechInstance.speak(sentence, queueMode) {
				Log.d(EyeAIApp.APP_LOG_TAG, "TTS finished final streaming chunk. Invoking completion callback.")
				isCurrentlyStreaming = false
				onStreamingComplete()
			}
		} else {
			textToSpeechInstance.speak(sentence, queueMode)
		}
	}

	suspend fun speakAndHandleUi(text: String) {
		val toSpeak = text.trim()
		if (toSpeak.isEmpty()) {
			onStreamingComplete()
			return
		}

		withContext(Dispatchers.Main) { llmResponseText?.text = eyeAIApp.getString(R.string.llm_response, toSpeak) }
		textToSpeechInstance.speak(toSpeak) {
			Log.d(EyeAIApp.APP_LOG_TAG, "TTS finished non-streaming response. Invoking completion callback.")
			onStreamingComplete()
		}
	}
}
