package com.algorithmic_alliance.eyeaiapp.llm

import android.annotation.SuppressLint
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
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

	private val endpoint =
		customEndpoint ?: GOOGLE_GEN_AI_ENDPOINT

	override suspend fun generate(prompt: String): String {
		var connection: HttpsURLConnection? = null
		var reader: BufferedReader? = null

		try {
			val url = URL("$endpoint/v1beta/models/$MODEL_NAME:generateContent?key=$apiKey")

			connection = url.openConnection() as HttpsURLConnection
			connection.requestMethod = "POST"
			connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
			connection.setRequestProperty("Accept", "application/json")
			connection.doOutput = true


			if (customEndpoint != null) {
				// TODO: do not allow this when adding a production build variant!
				Log.w(
					EyeAIApp.APP_LOG_TAG,
					"Disabling certificate verification since we are using a custom google ai studio endpoint that points to the MockGoogleLLMServer to allow self signed certificates"
				)
				connection.hostnameVerifier = HostnameVerifier { _, _ -> true }
				connection.sslSocketFactory = createTrustAllSslSocketFactory()
			}

			val requestBody = createRequestBody(prompt)
			val outputStream: OutputStream = connection.outputStream
			outputStream.write(requestBody.toByteArray(Charsets.UTF_8))
			outputStream.close()

			val responseCode = connection.responseCode
			if (responseCode != HttpsURLConnection.HTTP_OK) {
				val errorStream = connection.errorStream
				reader = BufferedReader(InputStreamReader(errorStream))
				val errorResponse = reader.readText()
				throw RuntimeException("API request failed: $responseCode - $errorResponse")
			}

			reader = BufferedReader(InputStreamReader(connection.inputStream))
			val response = reader.readText()

			return parseResponse(response)
		} catch (e: Exception) {
			val errorMsg = "Error in LLM generate: ${e.message}"
			Log.e(EyeAIApp.APP_LOG_TAG, errorMsg, e)
			return errorMsg
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
