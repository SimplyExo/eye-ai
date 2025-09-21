package com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio

import android.annotation.SuppressLint
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import javax.net.ssl.*

class NetworkClient(
	private val endpoint: String,
	private val apiKey: String,
	private val customEndpoint: String?
) {

	fun postJson(urlPath: String, requestBody: String, acceptStream: Boolean = false): BufferedReader {
		val urlStr = "$endpoint$urlPath?key=$apiKey"
		Log.d(EyeAIApp.APP_LOG_TAG, "HTTP POST${if (acceptStream) " STREAM" else ""} to $urlStr")

		val url = URL(urlStr)
		val connection = url.openConnection() as HttpsURLConnection

		configureConnection(connection, acceptStream)
		sendRequest(connection, requestBody)
		validateResponse(connection)

		return BufferedReader(InputStreamReader(connection.inputStream))
	}

	private fun configureConnection(connection: HttpsURLConnection, acceptStream: Boolean) {
		connection.requestMethod = "POST"
		connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
		connection.doOutput = true

		if (acceptStream) {
			connection.setRequestProperty("Accept", "text/event-stream")
			connection.readTimeout = 0
		}

		if (customEndpoint != null) {
			Log.w(EyeAIApp.APP_LOG_TAG, "Custom endpoint: disabling hostname/cert checks for dev-only")
			connection.hostnameVerifier = HostnameVerifier { _, _ -> true }
			connection.sslSocketFactory = createTrustAllSslSocketFactory()
		}
	}

	private fun sendRequest(connection: HttpsURLConnection, requestBody: String) {
		connection.outputStream.use {
			it.write(requestBody.toByteArray(Charsets.UTF_8))
		}
	}

	private fun validateResponse(connection: HttpsURLConnection) {
		val responseCode = connection.responseCode
		if (responseCode != HttpURLConnection.HTTP_OK) {
			val errorResponse = connection.errorStream?.bufferedReader()?.readText() ?: "No error body"
			throw GeminiErrorHandler.handleHttpError(responseCode, errorResponse)
		}
	}

	@SuppressLint("TrustAllX509TrustManager", "CustomX509TrustManager")
	private fun createTrustAllSslSocketFactory(): SSLSocketFactory {
		val trustAllCerts = arrayOf(
			object : X509TrustManager {
				override fun checkClientTrusted(chain: Array<java.security.cert.X509Certificate>, authType: String) {}
				override fun checkServerTrusted(chain: Array<java.security.cert.X509Certificate>, authType: String) {}
				override fun getAcceptedIssuers(): Array<java.security.cert.X509Certificate> = arrayOf()
			}
		)
		val sslContext = SSLContext.getInstance("TLS")
		sslContext.init(null, trustAllCerts, java.security.SecureRandom())
		return sslContext.socketFactory
	}
}
