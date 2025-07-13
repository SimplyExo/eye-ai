package com.algorithmic_alliance.eyeaiapp.llm

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL

/** REST API client for google ai studio (generative ai). Not using google's maven central genai library, as it has a critical bug. */
class GoogleAIStudioLLM(private val apiKey: String) : LLM {
	companion object {
		const val MODEL_NAME: String = "gemini-2.5-flash-preview-05-20"
		private const val BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
	}

	override suspend fun generate(command: String): String {
		var connection: HttpURLConnection? = null
		var reader: BufferedReader? = null

		try {
			val url = URL("${BASE_URL}${MODEL_NAME}:generateContent?key=${apiKey}")

			connection = url.openConnection() as HttpURLConnection
			connection.requestMethod = "POST"
			connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
			connection.setRequestProperty("Accept", "application/json")
			connection.doOutput = true

			val requestBody = createRequestBody(command)
			val outputStream: OutputStream = connection.outputStream
			outputStream.write(requestBody.toByteArray(Charsets.UTF_8))
			outputStream.close()

			val responseCode = connection.responseCode
			if (responseCode != HttpURLConnection.HTTP_OK) {
				val errorStream = connection.errorStream
				reader = BufferedReader(InputStreamReader(errorStream))
				val errorResponse = reader.readText()
				throw RuntimeException("API request failed: $responseCode - $errorResponse")
			}

			reader = BufferedReader(InputStreamReader(connection.inputStream))
			val response = reader.readText()

			return parseResponse(response)
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Error in LLM generate: ${e.message}", e)
			throw RuntimeException("Failed to generate content: ${e.message}", e)
		} finally {
			reader?.close()
			connection?.disconnect()
		}
	}

	private fun createRequestBody(prompt: String): String {
		return JSONObject().apply {
			put("systemInstruction", JSONObject().apply {
				put("parts", JSONArray().apply {
					put(JSONObject().apply {
						put("text", LLM.SYSTEM_PROMPT)
					})
				})
			})
			put("contents", JSONArray().apply {
				put(JSONObject().apply {
					put("parts", JSONArray().apply {
						put(JSONObject().apply {
							put("text", prompt)
						})
					})
				})
			})
			put("generationConfig", JSONObject().apply {
				put("thinkingConfig", JSONObject().apply {
					put("thinkingBudget", 0) // disables thinking mode
				})
			})
		}.toString()
	}

	private fun parseResponse(responseBody: String): String {
		val jsonResponse = JSONObject(responseBody)
		val candidates = jsonResponse.getJSONArray("candidates")
		if (candidates.length() == 0) {
			throw RuntimeException("No candidates in response")
		}

		val firstCandidate = candidates.getJSONObject(0)
		val content = firstCandidate.getJSONObject("content")
		val parts = content.getJSONArray("parts")
		if (parts.length() == 0) {
			throw RuntimeException("No parts in candidate content")
		}

		return parts.getJSONObject(0).getString("text")
	}
}