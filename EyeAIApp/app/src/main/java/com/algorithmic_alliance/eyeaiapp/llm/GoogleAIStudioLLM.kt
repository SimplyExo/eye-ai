package com.algorithmic_alliance.eyeaiapp.llm

import android.annotation.SuppressLint
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.SSLSocketFactory
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager

class GoogleAIStudioLLM(private val apiKey: String, private val customEndpoint: String?) : LLM {
	companion object {
		const val MODEL_NAME: String = "gemini-2.5-flash-lite"
		private const val GOOGLE_GEN_AI_ENDPOINT = "https://generativelanguage.googleapis.com"
	}

	private val endpoint = if (customEndpoint.isNullOrEmpty()) GOOGLE_GEN_AI_ENDPOINT else customEndpoint

	private fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000 //for debugging only.

	override fun generate(command: String, structured: Boolean): String {
		var connection: HttpsURLConnection? = null
		var reader: BufferedReader? = null
		val totalStart = System.nanoTime()

		try {
			val urlStr = "$endpoint/v1beta/models/$MODEL_NAME:generateContent?key=$apiKey"
			Log.d(EyeAIApp.APP_LOG_TAG, "HTTP POST to $urlStr (structured=$structured)")

			val url = URL(urlStr)
			connection = url.openConnection() as HttpsURLConnection
			connection.requestMethod = "POST"
			connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
			connection.doOutput = true

			if (customEndpoint != null) {
				Log.w(EyeAIApp.APP_LOG_TAG, "Custom endpoint: disabling hostname/cert checks for dev-only")
				connection.hostnameVerifier = HostnameVerifier { _, _ -> true }
				connection.sslSocketFactory = createTrustAllSslSocketFactory()
			}

			val requestBody = createRequestBody(command, structured)
			connection.outputStream.use { it.write(requestBody.toString().toByteArray(Charsets.UTF_8)) }

			val responseCode = connection.responseCode
			if (responseCode != HttpURLConnection.HTTP_OK) {
				val errorResponse = connection.errorStream?.bufferedReader()?.readText() ?: "No error body"
				Log.e(EyeAIApp.APP_LOG_TAG, "HTTP error body: ${errorResponse.take(1000)}")
				throw RuntimeException("API request failed: $responseCode - $errorResponse")
			}

			reader = BufferedReader(InputStreamReader(connection.inputStream))
			val response = reader.readText()
			val parsed = parseResponse(response)

			Log.d(EyeAIApp.APP_LOG_TAG, "Total LLM HTTP roundtrip: ${elapsedMs(totalStart)} ms")
			return parsed
		} catch (e: Exception) {
			val dur = elapsedMs(totalStart)
			val errorMsg = "Error in LLM generate after $dur ms: ${e.message}"
			Log.e(EyeAIApp.APP_LOG_TAG, errorMsg, e)
			return "Fehler bei der Anfrage: ${e.message}"
		} finally {
			reader?.close()
			connection?.disconnect()
		}
	}

	fun generateStream(
		command: String,
		onChunk: (String) -> Unit,
		onComplete: () -> Unit,
		onError: (Exception) -> Unit
	) {
		CoroutineScope(Dispatchers.IO).launch {
			var connection: HttpsURLConnection? = null
			var reader: BufferedReader? = null
			val totalStart = System.nanoTime()
			var hasCalledComplete = false

			//onComplete needed to check whether
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
				val urlStr = "$endpoint/v1beta/models/$MODEL_NAME:streamGenerateContent?key=$apiKey"
				Log.d(EyeAIApp.APP_LOG_TAG, "HTTP POST STREAM to $urlStr")

				val url = URL(urlStr)
				connection = url.openConnection() as HttpsURLConnection
				connection.requestMethod = "POST"
				connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
				connection.setRequestProperty("Accept", "text/event-stream")
				connection.doOutput = true
				connection.readTimeout = 0

				if (customEndpoint != null) {
					Log.w(EyeAIApp.APP_LOG_TAG, "Custom endpoint: disabling hostname/cert checks for dev-only")
					connection.hostnameVerifier = HostnameVerifier { _, _ -> true }
					connection.sslSocketFactory = createTrustAllSslSocketFactory()
				}

				val requestBody = createRequestBody(command, structured = false)
				connection.outputStream.use { it.write(requestBody.toString().toByteArray(Charsets.UTF_8)) }

				val responseCode = connection.responseCode
				if (responseCode != HttpURLConnection.HTTP_OK) {
					val errorResponse = connection.errorStream?.bufferedReader()?.readText() ?: "No error body"
					throw RuntimeException("API request failed: $responseCode - $errorResponse")
				}

				reader = BufferedReader(InputStreamReader(connection.inputStream))

				val eventBuffer = StringBuilder()
				val rawBuffer = StringBuilder()

				fun extractObjectsFromRawBuffer() {
					var buf = rawBuffer.toString()
					var idx = 0
					while (true) {
						val start = buf.indexOf('{', idx)
						if (start == -1) break
						var brace = 0
						var end: Int = -1
						var i = start
						while (i < buf.length) {
							val ch = buf[i]
							if (ch == '{') brace++
							else if (ch == '}') {
								brace--
								if (brace == 0) { end = i; break }
							}
							i++
						}
						if (end == -1) break
						val objStr = buf.substring(start, end + 1)
						try {
							Log.d(EyeAIApp.APP_LOG_TAG, "Found complete JSON object in buffer (len=${objStr.length}), passing to parser.")
							parseStreamChunk(objStr, onChunk)
						} catch (e: Exception) {
							Log.w(EyeAIApp.APP_LOG_TAG, "parseStreamChunk threw for extracted object; continuing.", e)
						}
						buf = buf.substring(end + 1)
						idx = 0
					}
					rawBuffer.setLength(0)
					rawBuffer.append(buf)
				}

				try {
					reader.forEachLine { rawLine ->
						if (hasCalledComplete) return@forEachLine // Stop processing if already completed

						Log.d(EyeAIApp.APP_LOG_TAG, "Raw stream line: $rawLine")
						val line = rawLine.trimEnd('\r', '\n')

						if (line.startsWith("data:")) {
							val payload = line.substringAfter("data:").trimStart()
							if (payload == "[DONE]" || payload == "\"[DONE]\"") {
								Log.d(EyeAIApp.APP_LOG_TAG, "Stream signalled DONE")
								if (eventBuffer.isNotEmpty()) {
									val evt = eventBuffer.toString()
									try { parseStreamChunk(evt, onChunk) } catch (e: Exception) {
										Log.w(EyeAIApp.APP_LOG_TAG, "Failed final SSE event before DONE", e)
									}
									eventBuffer.clear()
								}
								safeCallComplete()
								return@forEachLine
							} else {
								eventBuffer.append(payload)
								if (eventBuffer.isNotEmpty()) {
									rawBuffer.append(eventBuffer.toString())
									eventBuffer.clear()
									extractObjectsFromRawBuffer()
								}
								return@forEachLine
							}
						}

						if (line.isBlank()) {
							if (eventBuffer.isNotEmpty()) {
								rawBuffer.append(eventBuffer.toString())
								eventBuffer.clear()
								extractObjectsFromRawBuffer()
							}
							return@forEachLine
						}

						val trimmed = line.trim()
						if (trimmed == "[" || trimmed == "," || trimmed == "]") {
							rawBuffer.append(trimmed)
							extractObjectsFromRawBuffer()
							return@forEachLine
						}

						rawBuffer.append(line)
						extractObjectsFromRawBuffer()
					}
				} catch (e: Exception) {
					if (!hasCalledComplete) {
						Log.w(EyeAIApp.APP_LOG_TAG, "Exception during stream reading", e)
						safeCallError(e)
						return@launch
					}
				}
				try {
					if (!hasCalledComplete) {
						if (eventBuffer.isNotEmpty()) {
							rawBuffer.append(eventBuffer.toString())
							eventBuffer.clear()
						}
						extractObjectsFromRawBuffer()
						val remainder = rawBuffer.toString().trim()
						if (remainder.isNotEmpty()) {
							if (remainder.startsWith("[") && remainder.endsWith("]")) {
								try {
									Log.d(EyeAIApp.APP_LOG_TAG, "EOF: parsing final array remainder (len=${remainder.length})")
									parseStreamChunk(remainder, onChunk)
								} catch (e: Exception) {
									Log.w(EyeAIApp.APP_LOG_TAG, "EOF parse of remaining array failed", e)
								}
							} else {
								Log.d(EyeAIApp.APP_LOG_TAG, "EOF: leftover buffer after extraction (len=${remainder.length}).")
							}
						}

						// Stream beendet - explizites Completion-Signal
						Log.d(EyeAIApp.APP_LOG_TAG, "Stream finished naturally - calling completion")
						safeCallComplete()
					}
				} catch (e: Exception) {
					Log.w(EyeAIApp.APP_LOG_TAG, "Failed to parse buffered final event on EOF", e)
					if (!hasCalledComplete) {
						safeCallComplete() // Trotz Parsing-Fehler als beendet markieren
					}
				}

			} catch (e: Exception) {
				safeCallError(e)
			} finally {
				try {
					reader?.close()
					connection?.disconnect()
				} catch (e: Exception) {
					Log.w(EyeAIApp.APP_LOG_TAG, "Error closing stream resources", e)
				}
			}
		}
	}


	//Parsing the stream chunk
	private fun parseStreamChunk(rawJsonCandidate: String, onChunk: (String) -> Unit) {
		try {
			Log.d(EyeAIApp.APP_LOG_TAG, "Attempting to parse JSON chunk (len=${rawJsonCandidate.length})...")
			val trimmed = rawJsonCandidate.trim()
			val root: Any = try {
				if (trimmed.startsWith("[")) JSONArray(trimmed) else JSONObject(trimmed)
			} catch (e: JSONException) {
				val firstObj = rawJsonCandidate.indexOf('{')
				val lastObj = rawJsonCandidate.lastIndexOf('}')
				if (firstObj != -1 && lastObj != -1 && lastObj > firstObj) {
					JSONObject(rawJsonCandidate.substring(firstObj, lastObj + 1))
				} else {
					val firstArr = rawJsonCandidate.indexOf('[')
					val lastArr = rawJsonCandidate.lastIndexOf(']')
					if (firstArr != -1 && lastArr != -1 && lastArr > firstArr) {
						JSONArray(rawJsonCandidate.substring(firstArr, lastArr + 1))
					} else {
						Log.w(EyeAIApp.APP_LOG_TAG, "Could not coerce chunk to JSON: '${rawJsonCandidate.take(200)}...'")
						return
					}
				}
			}

			fun normalizeText(s: String): String {
				return s.replace("\r", " ").replace("\n", " ").replace(Regex("\\s+"), " ").trim()
			}

			val found = LinkedHashSet<String>()

			fun processCandidate(cand: JSONObject) {
				val content = cand.optJSONObject("content")
				if (content != null) {
					val parts = content.optJSONArray("parts")
					if (parts != null) {
						for (i in 0 until parts.length()) {
							val pObj = parts.optJSONObject(i)
							val raw = if (pObj != null) pObj.optString("text", "") else parts.optString(i, "")
							val txt = normalizeText(raw)
							if (txt.isNotEmpty()) found.add(txt)
						}
					}
					val direct = normalizeText(content.optString("text", ""))
					if (direct.isNotEmpty()) found.add(direct)
				}
				val candText = normalizeText(cand.optString("text", ""))
				if (candText.isNotEmpty()) found.add(candText)
			}

			fun deepScan(node: Any?) {
				when (node) {
					is JSONObject -> {
						if (node.has("candidates")) {
							val cands = node.optJSONArray("candidates")
							if (cands != null) {
								for (i in 0 until cands.length()) {
									val cand = cands.opt(i)
									if (cand is JSONObject) processCandidate(cand)
								}
							}
						}
						if (node.has("parts")) {
							val parts = node.optJSONArray("parts")
							if (parts != null) {
								for (i in 0 until parts.length()) {
									val pObj = parts.optJSONObject(i)
									val raw = if (pObj != null) pObj.optString("text", "") else parts.optString(i, "")
									val txt = normalizeText(raw)
									if (txt.isNotEmpty()) found.add(txt)
								}
							}
						}
						val keys = node.keys()
						while (keys.hasNext()) {
							val k = keys.next()
							deepScan(node.opt(k))
						}
					}
					is JSONArray -> {
						for (i in 0 until node.length()) deepScan(node.opt(i))
					}
					else -> {
						// ignore primitives
					}
				}
			}

			deepScan(root)

			for (s in found) {
				if (s.length < 2) continue
				Log.d(EyeAIApp.APP_LOG_TAG, "Parsed text (normalized & deduped): '${s.take(200)}...'")
				onChunk(s)
			}

			Log.d(EyeAIApp.APP_LOG_TAG, "Finished parseStreamChunk.")
		} catch (e: JSONException) {
			Log.w(EyeAIApp.APP_LOG_TAG, "Could not parse stream chunk as JSON: '${rawJsonCandidate.take(200)}...'", e)
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Unexpected error while parsing stream chunk", e)
		}
	}

	private fun createRequestBody(prompt: String, structured: Boolean): JSONObject {
		val defaultResponseBody = JSONObject().apply {
			put("systemInstruction", JSONObject().apply {
				put("parts", JSONArray().apply {
					put(JSONObject().apply { put("text", LLM.SYSTEM_PROMPT) })
				})
			})
			put("contents", JSONArray().apply {
				put(JSONObject().apply {
					put("parts", JSONArray().apply {
						put(JSONObject().apply { put("text", prompt) })
					})
				})
			})
		}

		if (structured) {
			val schema = JSONObject().apply {
				put("type", "OBJECT")
				put("properties", JSONObject().apply {
					// new option to avoid another generation request
					put("interaction_text", JSONObject().apply {
						put("type", "STRING")
					})

					put("setting_intent", JSONObject().apply {
						put("type", "STRING")
						put("enum", JSONArray().apply {
							put("tts_speed"); put("voice"); put("leave"); put("none")
						})
					})
					put("requested_functions", JSONObject().apply {
						put("type", "OBJECT")
						put("properties", JSONObject().apply {
							put("einstellungen", JSONObject().apply { put("type", "BOOLEAN") })
							put("texterkennung", JSONObject().apply { put("type", "BOOLEAN") })
						})
					})
					put("changed_settings", JSONObject().apply {
						put("type", "ARRAY")
						put("items", JSONObject().apply {
							put("type", "OBJECT")
							put("properties", JSONObject().apply {
								put("tts_speed", JSONObject().apply { put("type", "NUMBER") })
								put("voice", JSONObject().apply { put("type", "NUMBER") })
								put("leave", JSONObject().apply { put("type", "BOOLEAN") })
							})
						})
					})
					put("approval", JSONObject().apply { put("type", "NUMBER") })
				})
			}

			val generationConfig = JSONObject().apply {
				put("response_mime_type", "application/json")
				put("response_schema", schema)
			}

			return defaultResponseBody.put("generationConfig", generationConfig)
		} else {
			return defaultResponseBody.put("generationConfig", JSONObject().apply {
				// unstrukturierte Antwort (z.B. für stream)
			})
		}
	}

	private fun parseResponse(responseBody: String): String {
		try {
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
			return parts.getJSONObject(0).getString("text")
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Failed to parse LLM JSON response", e)
			return responseBody
		}
	}
}

@SuppressLint("TrustAllX509TrustManager", "CustomX509TrustManager")
private fun createTrustAllSslSocketFactory(): SSLSocketFactory {
	val trustAllCerts = arrayOf<TrustManager>(
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