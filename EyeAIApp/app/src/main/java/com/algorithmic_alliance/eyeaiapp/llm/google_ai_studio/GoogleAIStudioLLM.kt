package com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.llm.LLM
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONException
import org.json.JSONObject

class GoogleAIStudioLLM(apiKey: String, customEndpoint: String?) : LLM {


	// handling to stop streaming
	@Volatile
	private var shouldStopStream = false

	fun stopCurrentStream() {
		shouldStopStream = true
	}

	companion object {
		const val MODEL_NAME: String = "gemini-2.5-flash-lite"
		private const val GOOGLE_GEN_AI_ENDPOINT = "https://generativelanguage.googleapis.com"

		private fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000
	}

	private val endpoint = if (customEndpoint.isNullOrEmpty()) GOOGLE_GEN_AI_ENDPOINT else customEndpoint
	private val networkClient = NetworkClient(endpoint, apiKey, customEndpoint)

	override fun generate(command: String, structured: Boolean): String {
		val totalStart = System.nanoTime()
		var reader: java.io.BufferedReader? = null

		return try {
			val requestBody = RequestBuilder.createRequestBody(command, structured).toString()
			reader = networkClient.postJson("/v1beta/models/$MODEL_NAME:generateContent", requestBody)
			val response = reader.readText()
			val parsed = parseResponse(response)

			Log.d(EyeAIApp.APP_LOG_TAG, "Total LLM HTTP roundtrip: ${elapsedMs(totalStart)} ms")
			parsed

		} catch (e: GeminiApiExceptionHandler) {
			val duration = elapsedMs(totalStart)
			Log.e(EyeAIApp.APP_LOG_TAG, "Gemini API error after $duration ms: ${e.userMessage}", e)
			e.userMessage
		} catch (e: Exception) {
			val duration = elapsedMs(totalStart)
			val errorMsg = "Error in LLM generate after $duration ms: ${e.message}"
			Log.e(EyeAIApp.APP_LOG_TAG, errorMsg, e)
			"Fehler bei der Anfrage: ${e.message}"
		} finally {
			reader?.close()
		}
	}

	fun generateStream(
		command: String,
		onChunk: (String) -> Unit,
		onComplete: () -> Unit,
		onError: (Exception) -> Unit
	) {

		shouldStopStream = false

		CoroutineScope(Dispatchers.IO).launch {
			val totalStart = System.nanoTime()
			var hasCalledComplete = false

			fun safeCallComplete() {
				synchronized(this@GoogleAIStudioLLM) {
					if (!hasCalledComplete) {
						hasCalledComplete = true
						Log.d(EyeAIApp.APP_LOG_TAG, "Stream completion signalled after ${elapsedMs(totalStart)} ms")
						onComplete()
					}
				}
			}

			fun safeCallError(e: Exception) {
				synchronized(this@GoogleAIStudioLLM) {
					if (!hasCalledComplete) {
						hasCalledComplete = true
						Log.e(EyeAIApp.APP_LOG_TAG, "Stream error after ${elapsedMs(totalStart)} ms", e)
						onError(e)
					}
				}
			}

			try {
				val requestBody = RequestBuilder.createRequestBody(command, structured = false).toString()
				val reader = networkClient.postJson("/v1beta/models/$MODEL_NAME:streamGenerateContent", requestBody, acceptStream = true)

				val parser = StreamParser(onChunk)
				val processor = StreamProcessor(parser)

				processor.processStream(
					reader = reader,
					onComplete = { safeCallComplete() },
					onError = { safeCallError(it) },
					hasCalledComplete = { hasCalledComplete || shouldStopStream}
				)

			} catch (e: GeminiApiExceptionHandler) {
				// Specifically for GeminiAPI Errors.
				if (!shouldStopStream) {
					Log.e(EyeAIApp.APP_LOG_TAG, "Gemini API streaming error ${e.errorCode}: ${e.userMessage}")
					safeCallError(e)
				}
			} catch (e: Exception) {
				if(!shouldStopStream) safeCallError(e)
			}
		}
	}

	private fun parseResponse(responseBody: String): String {
		return try {
			val jsonResponse = JSONObject(responseBody)
			val candidates = jsonResponse.getJSONArray("candidates")

			if (candidates.length() == 0) {
				Log.w(EyeAIApp.APP_LOG_TAG, "No candidates in LLM response.")
				return ""
			}

			val firstCandidate = candidates.getJSONObject(0)
			val content = firstCandidate.getJSONObject("content")
			val parts = content.getJSONArray("parts")

			if (parts.length() == 0) {
				Log.w(EyeAIApp.APP_LOG_TAG, "No parts in candidate content.")
				return ""
			}

			parts.getJSONObject(0).getString("text")

		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Failed to parse LLM JSON response", e)
			responseBody
		}
	}
}
