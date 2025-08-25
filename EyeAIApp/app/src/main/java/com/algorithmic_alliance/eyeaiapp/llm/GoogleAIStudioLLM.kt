package com.algorithmic_alliance.eyeaiapp.llm

import android.annotation.SuppressLint
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.SSLSocketFactory
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager

/**
 * REST API client for google ai studio (generative ai). Not using google's maven central genai library, as it has a critical bug.
 * @param customEndpoint if null, [GOOGLE_GEN_AI_ENDPOINT] is used, else [customEndpoint] for the endpoint
 */
class GoogleAIStudioLLM(private val apiKey: String, private val customEndpoint: String?) : LLM {
	companion object {
		const val MODEL_NAME: String = "gemini-2.5-flash-preview-05-20"
		private const val GOOGLE_GEN_AI_ENDPOINT = "https://generativelanguage.googleapis.com"
	}

	private val endpoint = if (customEndpoint == null || customEndpoint.isEmpty()) {
		GOOGLE_GEN_AI_ENDPOINT
	} else {
		customEndpoint
	}

	fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

	override fun generate(command: String, structured: Boolean): String {
		var connection: HttpsURLConnection? = null
		var reader: BufferedReader? = null
		val totalStart = System.nanoTime()

		try {
			val urlStr = "$endpoint/v1beta/models/$MODEL_NAME:generateContent?key=$apiKey"
			Log.d(EyeAIApp.APP_LOG_TAG, "HTTP POST to $urlStr (structured=$structured)")

			val url = URL(urlStr)
			val connOpenStart = System.nanoTime()
			connection = url.openConnection() as HttpsURLConnection
			val connOpenMs = elapsedMs(connOpenStart)
			Log.d(EyeAIApp.APP_LOG_TAG, "Connection opened in ${connOpenMs} ms")

			connection.requestMethod = "POST"
			connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
			connection.setRequestProperty("Accept", "application/json")
			connection.doOutput = true

			if (customEndpoint != null) {
				Log.w(EyeAIApp.APP_LOG_TAG, "Custom endpoint: disabling hostname/cert checks for dev-only")
				connection.hostnameVerifier = HostnameVerifier { _, _ -> true }
				connection.sslSocketFactory = createTrustAllSslSocketFactory()
			}

			val requestBody = createRequestBody(command, structured)
			val writeStart = System.nanoTime()
			val outputStream: OutputStream = connection.outputStream
			outputStream.write(requestBody.toString().toByteArray(Charsets.UTF_8))
			outputStream.close()
			val writeMs = elapsedMs(writeStart)
			Log.d(EyeAIApp.APP_LOG_TAG, "Wrote request body in ${writeMs} ms (size=${requestBody.toString().length} chars)")

			val responseCodeStart = System.nanoTime()
			val responseCode = connection.responseCode
			val responseCodeMs = elapsedMs(responseCodeStart)
			Log.d(EyeAIApp.APP_LOG_TAG, "Got responseCode=$responseCode after ${responseCodeMs} ms")

			if (responseCode != HttpURLConnection.HTTP_OK) {
				val errorStream = connection.errorStream
				reader = BufferedReader(InputStreamReader(errorStream))
				val errorResponse = reader.readText()
				Log.e(EyeAIApp.APP_LOG_TAG, "HTTP error body: ${errorResponse.take(1000)}")
				throw RuntimeException("API request failed: $responseCode - $errorResponse")
			}

			val readStart = System.nanoTime()
			reader = BufferedReader(InputStreamReader(connection.inputStream))
			val response = reader.readText()
			val readMs = elapsedMs(readStart)
			Log.d(EyeAIApp.APP_LOG_TAG, "Read response in ${readMs} ms (size=${response.length} chars)")

			val parseStart = System.nanoTime()
			val parsed = parseResponse(response)
			val parseMs = elapsedMs(parseStart)
			Log.d(EyeAIApp.APP_LOG_TAG, "Parsed response in ${parseMs} ms")

			Log.d(EyeAIApp.APP_LOG_TAG, "Total LLM HTTP roundtrip: ${elapsedMs(totalStart)} ms")
			return parsed
		} catch (e: Exception) {
			val dur = elapsedMs(totalStart)
			val errorMsg = "Error in LLM generate after ${dur} ms: ${e.message}"
			Log.e(EyeAIApp.APP_LOG_TAG, errorMsg, e)
			return errorMsg
		} finally {
			reader?.close()
			connection?.disconnect()
		}
	}

	private fun createRequestBody(prompt: String, structured: Boolean): JSONObject {
		val defaultResponseBody = JSONObject().apply {
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
		}

		if (structured) {
			val schema = JSONObject().apply {
				put("type", "OBJECT")
				put("properties", JSONObject().apply {

					//New option for requested functions, provides safer function calling
					put("requested_functions", JSONObject().apply {
						put("type", "OBJECT")
						put("description", "Identifies which core function the user wants to trigger.")
						put("properties", JSONObject().apply {

							put("einstellungen", JSONObject().apply {
								put("type", "BOOLEAN")
								put("description", "Set to true if the user wants to open or modify settings.")
							})

							put("texterkennung", JSONObject().apply {
								put("type", "BOOLEAN")
								put("description", "Set to true if the user wants to use the text recognition (OCR) feature.")
							})
						})
					})


					put("changed_settings", JSONObject().apply {
						put("type", "ARRAY")
						put("items", JSONObject().apply {
							put("type", "OBJECT")
							put("properties", JSONObject().apply {

								put("tts_speed", JSONObject().apply {
									put("type", "NUMBER")
									put(
										"description",
										"The new text-to-speech speed, e.g. 1.0, 1.5, or 0.8"
									)
								})

								put("voice", JSONObject().apply {
									put("type", "NUMBER")
									put(
										"description",
										"The new voice. If the user suggests it should be female, answer with 0. If male, answer with 1."
									)
								})

								put("leave", JSONObject().apply {
									put("type", "BOOLEAN")
									put("description", "Set to true if the user wants to leave the settings menu.")
								})
							})
						})
					})


					put("approval", JSONObject().apply {
						put("type", "NUMBER")
						put(
							"description",
							"Whether the user approves the change or doesn't. Answer with either 1 or 0. '1' for approval, '0' for disagreement."
						)
					})
				})
			}

			val generationConfig = JSONObject().apply {
				put("response_mime_type", "application/json")
				put("response_schema", schema)
			}

			return defaultResponseBody.put("generationConfig", generationConfig)
		} else {
			return defaultResponseBody.put("generationConfig", JSONObject().apply {
				put("temperature", 1.0)
			})
		}
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


/// SHOULD ONLY BE USED WHEN USING THE MOCK GOOGLE GEN AI STUDIO ENDPOINT, AS IT USES A SELF SIGNED CERTIFICATE, NEVER ANYWHERE ELSE!
@SuppressLint("TrustAllX509TrustManager", "CustomX509TrustManager")
fun createTrustAllSslSocketFactory(): SSLSocketFactory {
	val trustAllCerts = arrayOf<TrustManager>(
		object : X509TrustManager {
			override fun checkClientTrusted(
				chain: Array<java.security.cert.X509Certificate>,
				authType: String
			) {
			}

			override fun checkServerTrusted(
				chain: Array<java.security.cert.X509Certificate>,
				authType: String
			) {
			}

			override fun getAcceptedIssuers(): Array<java.security.cert.X509Certificate> = arrayOf()
		}
	)

	val sslContext = SSLContext.getInstance("TLS")
	sslContext.init(null, trustAllCerts, java.security.SecureRandom())
	return sslContext.socketFactory
}