package com.algorithmic_alliance.eyeaiapp.llm

import android.util.Log
import android.widget.TextView
import com.algorithmic_alliance.eyeaiapp.AIModelData
import com.algorithmic_alliance.eyeaiapp.AIModelData.depthEstimationData
import com.algorithmic_alliance.eyeaiapp.AIModelData.objectDetectionBoxes
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.llm.LLM.Companion.knownObjectLabels
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import com.google.android.gms.common.api.Response
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.lang.StringBuilder

class StateMachine(
	private val eyeAIApp: EyeAIApp,
	private val textToSpeechInstance: TextToSpeechInstance,
	private var lastLlmJsonResponse: String?,
	private val llmResponseText: TextView?,
	private val onStreamingComplete: () -> Unit = {}
) {

	enum class RequestedFunction {
		TEXT_RECOGNITION, SETTINGS, NONE, OBJECT_DETECTION
	}

	enum class SettingIntent {
		TTS_SPEED, VOICE, LEAVE, NONE
	}

	data class DetectedObject(
		val label: String,
		val distance: Float,
		val height: Float,
		val width: Float,
		val x: Float,
		val y: Float
	)

	private val jsonParser = JsonParser()
	private val sentenceBuffer = StringBuilder()
	private var isFirstStreamChunk = true
	private var lastEmittedChunk: String? = null
	private val sentenceDelimiters = charArrayOf('.', '!', '?')

	@Volatile
	private var isCurrentlyStreaming = false

	fun isStreaming(): Boolean = isCurrentlyStreaming

	//needed for logging only
	private fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

	suspend fun handleIdle(final: String): StateUpdate {
		val jsonResponse = generateLlmResponse(final, true) ?: return StateUpdate(State.IDLE, null)
		Log.d(EyeAIApp.APP_LOG_TAG, "handleIdle called with: '$final', parsed function: ${jsonParser.parseRequestedFunction(jsonResponse)}")

		Log.d(EyeAIApp.APP_LOG_TAG, "=== LLM RESPONSE DEBUG ===")
		Log.d(EyeAIApp.APP_LOG_TAG, "User input: '$final'")
		Log.d(EyeAIApp.APP_LOG_TAG, "LLM JSON response: $jsonResponse")
		Log.d(EyeAIApp.APP_LOG_TAG, "Parsed function: ${jsonParser.parseRequestedFunction(jsonResponse)}")
		Log.d(EyeAIApp.APP_LOG_TAG, "Parsed object query: '${jsonParser.parseObjectQuery(jsonResponse)}'")
		Log.d(EyeAIApp.APP_LOG_TAG, "========================")

		return when (jsonParser.parseRequestedFunction(jsonResponse)) {
			RequestedFunction.OBJECT_DETECTION -> {
				// Objekterkennung tatsächlich ausführen
				handleObjectDetection(jsonResponse)
				StateUpdate(State.IDLE, null)
			}

			RequestedFunction.TEXT_RECOGNITION -> {
				val ocrLast = eyeAIApp.ocrModel.lastResult.trim()
				if (ocrLast.isEmpty()) {
					Log.d(EyeAIApp.APP_LOG_TAG, "No OCR text available — skipping LLM OCR flow.")
					speakAndHandleUi("Entschuldigung, es wurde kein Text erkannt.")
					return StateUpdate(State.IDLE, null)
				}

				val prompt = eyeAIApp.llm!!.buildOcrPrompt(ocrLast)
				if (prompt.trim().isEmpty()) {
					Log.w(EyeAIApp.APP_LOG_TAG, "OCR prompt is empty — skipping LLM call.")
					speakAndHandleUi("Entschuldigung, ich konnte keinen sinnvollen Text erkennen.")
					return StateUpdate(State.IDLE, null)
				}

				generateAndStreamLlmResponse(prompt)
				StateUpdate(State.IDLE, null)
			}
			RequestedFunction.SETTINGS -> {
				speakAndHandleUi(LLM.SNIPPET_SETTINGS)
				lastLlmJsonResponse = jsonResponse
				StateUpdate(State.SETTINGS_MENU, lastLlmJsonResponse)
			}
			RequestedFunction.NONE -> {
				val direct = jsonParser.parseInteractionText(jsonResponse)
				if (!direct.isNullOrBlank()) {
					// used to avoid double calling, uses text of JSON-Field "interaction_text" if possible
					speakAndHandleUi(direct)
					StateUpdate(State.IDLE, null)
				} else {
					// Fallback
					val fallbackResponse = generateLlmResponse(final, false) ?: jsonResponse
					speakAndHandleUi(fallbackResponse)
					StateUpdate(State.IDLE, null)
				}
			}
		}
	}

	suspend fun handleSettingsMenu(final: String): StateUpdate {
		val intentPrompt = """Der Nutzer ist im Einstellungsmenü und sagt: '$final'.
		Klassifiziere die Absicht des Nutzers in eine der folgenden Kategorien und gib sie im Feld 'setting_intent' zurück:
		- 'tts_speed': Wenn der Nutzer die Sprechgeschwindigkeit ändern will (z.B. "schneller sprechen").
		- 'voice': Wenn der Nutzer die Stimme des Assistenten ändern will (z.B. "Stimme ändern", "andere Stimme", "Assistentenagenten anpassen").
		- 'leave': Wenn der Nutzer die Einstellungen verlassen will.
		- 'none': Wenn keine der obigen Absichten klar erkennbar ist.
		Antworte NUR mit dem JSON-Objekt.
		Beispiel für die Eingabe "ich will eine andere Stimme": {"setting_intent": "voice"}
		Beispiel für die Eingabe "verlassen": {"setting_intent": "leave"}
		"""
		val jsonResponse = generateLlmResponse(intentPrompt, true) ?: return StateUpdate(State.SETTINGS_MENU, null)

		return when (jsonParser.parseSettingIntent(jsonResponse)) {

			SettingIntent.TTS_SPEED -> {
				lastLlmJsonResponse = jsonResponse
				speakAndHandleUi(LLM.SNIPPET_TTS_SPEED)
				StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
			}
			SettingIntent.VOICE -> {
				lastLlmJsonResponse = jsonResponse
				speakAndHandleUi(LLM.SNIPPET_VOICE)
				StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
			}
			SettingIntent.LEAVE -> {
				val syntheticLeave = JSONObject().apply {
					put("changed_settings", JSONArray().apply {
						put(JSONObject().apply {
							put("leave", true)
						})
					})
				}
				lastLlmJsonResponse = syntheticLeave.toString()
				speakAndHandleUi("Möchten Sie die Einstellungen wirklich verlassen?")
				StateUpdate(State.SETTINGS_ACTION, lastLlmJsonResponse)
			}
			SettingIntent.NONE -> {
				val response = "Ich habe das leider nicht verstanden. Sie können die Sprechgeschwindigkeit anpassen, die Stimme ändern oder die Einstellungen verlassen."
				speakAndHandleUi(response)
				StateUpdate(State.SETTINGS_MENU, lastLlmJsonResponse)
			}
		}
	}

	suspend fun handleSettingsChoice(final: String): StateUpdate {

		val currentIntent = lastLlmJsonResponse?.let { jsonParser.parseSettingIntent(it) }

		val prompt = if (currentIntent == SettingIntent.TTS_SPEED) {
			"Die aktuelle Sprechgeschwindigkeit ist ${textToSpeechInstance.speechRate}. Der Nutzer möchte folgendes: '$final'. Passe die Geschwindigkeit entsprechend an."
		} else {
			"Führe die folgende Aktion aus: '$final'."
		}
		val jsonResponse = generateLlmResponse(prompt, true) ?: return StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
		lastLlmJsonResponse = jsonResponse
		val confirmationQuestion = jsonParser.createConfirmationQuestion(jsonResponse)
		speakAndHandleUi(confirmationQuestion)
		return StateUpdate(State.SETTINGS_ACTION, jsonResponse)
	}

	suspend fun handleSettingsAction(final: String): StateUpdate {
		val prompt = "Würdest du sagen der Nutzer hat diesen Command bestätigt? Die Antwort des Nutzers war $final. Antworte bitte mit einer JSON-Antwort in approval."
		val jsonResponse = generateLlmResponse(prompt, true) ?: return StateUpdate(State.IDLE, lastLlmJsonResponse)

		if (jsonParser.isApproved(jsonResponse) && lastLlmJsonResponse != null) {
			val success = jsonParser.applySettings(lastLlmJsonResponse!!)
			if (!success) {
				speakAndHandleUi("Entschuldigung, beim Anwenden der Einstellung ist ein Fehler aufgetreten.")
			}
		} else {
			speakAndHandleUi("Okay, ich habe den Vorgang abgebrochen.")
		}
		return StateUpdate(State.IDLE, null)
	}

	private suspend fun handleObjectDetection(jsonResponse: String) {
		val specificQuery = jsonParser.parseObjectQuery(jsonResponse)?.lowercase()?.trim()

		//needs a specific object -> avoiding multiple requests and boilerplate code
		if (specificQuery.isNullOrEmpty()) {
			Log.w(EyeAIApp.APP_LOG_TAG, "Objekterkennung ohne spezifische Anfrage aufgerufen")
			speakAndHandleUi("Bitte nennen Sie ein spezifisches Objekt, nach dem Sie suchen möchten, zum Beispiel: 'Wo ist der Stuhl?'")
			return
		}

		val objectDetectionBoxes = objectDetectionBoxes.get()
		val depthEstimationData = depthEstimationData.get()

		Log.d(EyeAIApp.APP_LOG_TAG, "Searching for: '$specificQuery'")
		Log.d(EyeAIApp.APP_LOG_TAG, "Available objects: ${objectDetectionBoxes?.size}")
		Log.d(EyeAIApp.APP_LOG_TAG, "depth data: ${depthEstimationData?.size}")


		if (objectDetectionBoxes.isNullOrEmpty()) {
			speakAndHandleUi("Entschuldigung, ich konnte gerade keine Objekte erkennen.")
			return
		}

		if (depthEstimationData.isEmpty()) {
			speakAndHandleUi("Entschuldigung, die Tiefenerkennung ist derzeit nicht verfügbar.")
			return
		}

		val depthWidth = 256
		val depthHeight = 256

		if (depthEstimationData.size != depthWidth * depthHeight) {
			speakAndHandleUi("Entschuldigung, die Tiefendaten haben eine unerwartete Größe.")
			return
		}

		// List of all objects with depth
		val detectedObjects = objectDetectionBoxes.mapNotNull { box ->
			val label = box.clsName
			if (label in knownObjectLabels) {
				// Evaluating coordinates for getting the depth
				val depthX = (box.cx * (depthWidth - 1)).toInt().coerceIn(0, depthWidth - 1)
				val depthY = (box.cy * (depthHeight - 1)).toInt().coerceIn(0, depthHeight - 1)
				val depthIndex = depthY * depthWidth + depthX

				Log.d(EyeAIApp.APP_LOG_TAG, "Objekt '$label': Box(${box.cx}, ${box.cy}) -> Depth[$depthX, $depthY] = Index $depthIndex")

				val distance = if (depthIndex < depthEstimationData.size) {
					depthEstimationData[depthIndex]
				} else {
					Log.w(EyeAIApp.APP_LOG_TAG, "Depth-Index out of bounds")
					-1f
				}

				DetectedObject(label, distance, box.h, box.w, box.cx, box.cy)
			} else null
		}

		if (detectedObjects.isEmpty()) {
			speakAndHandleUi("Ich konnte leider keine bekannten Objekte erkennen.")
			return
		}

		//Specific object search
		val foundObject = detectedObjects.find { obj ->
			val objLabel = obj.label.lowercase()
			objLabel == specificQuery ||
				objLabel.contains(specificQuery) ||
				specificQuery.contains(objLabel)
		}

		if (foundObject != null) {
			Log.d(EyeAIApp.APP_LOG_TAG, "object found: ${foundObject.label} at ${foundObject.distance}m")

			//Generating a prompt for the specific object
			val prompt = eyeAIApp.llm!!.buildObjectDetectionPrompt(
				label = foundObject.label,
				height = foundObject.height,
				width = foundObject.width,
				x = foundObject.x,
				y = foundObject.y,
				distance = foundObject.distance
			)

			//Streaming the response
			generateAndStreamLlmResponse(prompt)
		} else {
			//Object not found
			val availableObjects = detectedObjects.map { it.label }.distinct().take(5).joinToString(", ")
			Log.d(EyeAIApp.APP_LOG_TAG, "Object '$specificQuery' not found. Available objects: $availableObjects")
			speakAndHandleUi("Entschuldigung, das Objekt '$specificQuery' konnte ich nicht finden. Ich sehe aber folgende Objekte: $availableObjects. Versuchen Sie es mit einem dieser Objekte.")
		}
	}

	private suspend fun generateLlmResponse(prompt: String, structured: Boolean): String? {
		val promptTrimmed = prompt.trim()
		if (promptTrimmed.isEmpty()) return null
		val start = System.nanoTime()
		return try {
			val result = eyeAIApp.llm!!.generate(promptTrimmed, structured)
			Log.d(EyeAIApp.APP_LOG_TAG, "LLM generate (non-stream) END duration=${elapsedMs(start)} ms")
			result
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "LLM generate (non-stream) EXCEPTION after ${elapsedMs(start)} ms", e)
			speakAndHandleUi("Entschuldigung, bei der Anfrage ist ein Fehler aufgetreten.")
			null
		}
	}

	//used for text-recognition only.
	private suspend fun generateAndStreamLlmResponse(prompt: String) {
		val llm = (eyeAIApp.llm as? GoogleAIStudioLLM) ?: run {
			Log.e(EyeAIApp.APP_LOG_TAG, "LLM instance is not GoogleAIStudioLLM, cannot stream.")
			speakAndHandleUi("Ein interner Fehler ist aufgetreten: Streaming nicht möglich.")
			onStreamingComplete()
			return
		}
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

	//buffering needed to make speaking the sentences more natural
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

	//freeing resources and invoking callback after the stream
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

	private suspend fun speakAndHandleUi(text: String) {
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

	//Parsing only.
	private inner class JsonParser {

		fun parseInteractionText(jsonString: String): String? {
			return try {
				val obj = JSONObject(jsonString)
				val txt = obj.optString("interaction_text", "").trim()
				if (txt.isNotEmpty()) return txt

				// Fallback: vielleicht hat die LLM-Antwort das Feld "text" direkt oder "message"
				val direct = obj.optString("text", "").trim()
				if (direct.isNotEmpty()) return direct

				null
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseInteractionText", e)
				null
			}
		}
		fun parseSettingIntent(jsonString: String): SettingIntent {
			return try {
				when (JSONObject(jsonString).optString("setting_intent", "none")) {
					"tts_speed" -> SettingIntent.TTS_SPEED
					"voice" -> SettingIntent.VOICE
					"leave" -> SettingIntent.LEAVE
					else -> SettingIntent.NONE
				}
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseSettingIntent", e)
				SettingIntent.NONE
			}
		}

		fun parseRequestedFunction(jsonString: String): RequestedFunction {
			return try {
				val requestedFunctions = JSONObject(jsonString).optJSONObject("requested_functions")
				when {
					requestedFunctions?.optBoolean("texterkennung", false) == true -> RequestedFunction.TEXT_RECOGNITION
					requestedFunctions?.optBoolean("einstellungen", false) == true -> RequestedFunction.SETTINGS
					requestedFunctions?.optBoolean("objekterkennung", false) == true -> RequestedFunction.OBJECT_DETECTION
					else -> RequestedFunction.NONE
				}
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseRequestedFunction", e)
				RequestedFunction.NONE
			}
		}

		fun parseObjectQuery(jsonString: String): String? {
			return try {
				val query = JSONObject(jsonString).optString("object_query", null)
				if (query.isNullOrBlank()) null else query
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseObjectQuery", e)
				null
			}
		}

		fun createConfirmationQuestion(jsonString: String): String {
			try {
				val changedSettings = JSONObject(jsonString).optJSONArray("changed_settings")
				if (changedSettings != null && changedSettings.length() > 0) {
					val firstChange = changedSettings.getJSONObject(0)
					when {
						firstChange.has("tts_speed") -> {
							val newSpeed = firstChange.getDouble("tts_speed")
							return "Verstanden. Soll ich die Sprachgeschwindigkeit auf $newSpeed setzen?"
						}
						firstChange.has("voice") -> {
							val voice = firstChange.getString("voice")
							return if (voice == "1") "Verstanden. Soll die Assistentenstimme nun weiblich sein?"
							else "Verstanden. Soll die Assistentenstimme nun männlich sein?"
						}
						firstChange.has("leave") -> return "Möchten Sie die Einstellungen wirklich verlassen?"
					}
				}
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in createConfirmationQuestion", e)
			}
			return "Soll ich die angeforderte Änderung durchführen?"
		}

		fun isApproved(jsonString: String): Boolean {
			return try {
				JSONObject(jsonString).optInt("approval", 0) == 1
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in isApproved", e)
				false
			}
		}

		suspend fun applySettings(jsonString: String): Boolean {
			try {
				val changedSettings = JSONObject(jsonString).getJSONArray("changed_settings")
				for (i in 0 until changedSettings.length()) {
					val setting = changedSettings.getJSONObject(i)
					when {
						setting.has("tts_speed") -> {
							val newSpeed = setting.getDouble("tts_speed").toFloat()
							textToSpeechInstance.setSpeechRate(newSpeed)
							Log.d(EyeAIApp.APP_LOG_TAG, "TTS-Geschwindigkeit wird auf $newSpeed gesetzt.")
							speakAndHandleUi("Die Einstellung wurde erfolgreich geändert.")
						}
						setting.has("voice") -> {
							val voice = setting.getInt("voice")
							textToSpeechInstance.setVoice(voice)
							Log.d(EyeAIApp.APP_LOG_TAG, "Stimme wird auf $voice gesetzt.")
							speakAndHandleUi("Die Einstellung wurde erfolgreich geändert.")
						}
						setting.has("leave") -> {
							speakAndHandleUi("Die Einstellungen wurden verlassen.")
						}
					}
				}
				return true
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "Fehler bei der Verarbeitung der JSON-Aktion.", e)
				return false
			}
		}
	}
}