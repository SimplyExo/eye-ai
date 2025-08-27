package com.algorithmic_alliance.eyeaiapp.llm

import android.util.Log
import android.widget.TextView
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

class StateMachine(
	private val eyeAIApp: EyeAIApp,
	private val textToSpeechInstance: TextToSpeechInstance,
	private var lastLlmJsonResponse: String?,
	private val llmResponseText: TextView?
) {

	// Enum class for the initial function request
	enum class RequestedFunction {
		TEXT_RECOGNITION, SETTINGS, NONE
	}

	// Enum class for intents within the settings menu
	enum class SettingIntent {
		TTS_SPEED, VOICE, LEAVE, NONE
	}

	// private parser
	private val jsonParser = JsonParser()

	private fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

	// Handling Idle state
	suspend fun handleIdle(final: String): StateUpdate {
		val jsonResponse = generateLlmResponse(final, true) ?: return StateUpdate(State.IDLE, null)

		return when (jsonParser.parseRequestedFunction(jsonResponse)) {
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

				val ocrResponse = generateLlmResponse(prompt, false)
				if (ocrResponse.isNullOrBlank()) {
					Log.w(EyeAIApp.APP_LOG_TAG, "OCR LLM returned null/empty response.")
					speakAndHandleUi("Entschuldigung, ich konnte keine passende Antwort zum erkannten Text generieren.")
				} else {
					speakAndHandleUi(ocrResponse)
				}
				StateUpdate(State.IDLE, null)
			}
			RequestedFunction.SETTINGS -> {
				speakAndHandleUi(LLM.SNIPPET_SETTINGS)
				lastLlmJsonResponse = jsonResponse
				StateUpdate(State.SETTINGS_MENU, lastLlmJsonResponse)
			}
			RequestedFunction.NONE -> {
				val fallbackResponse = generateLlmResponse(final, false) ?: jsonResponse
				speakAndHandleUi(fallbackResponse)
				StateUpdate(State.IDLE, null)
			}
		}
	}

	// Handling the user choice while in SETTINGS_MENU
	suspend fun handleSettingsMenu(final: String): StateUpdate {
		// The prompt was adapted to be more precise than before, allowing the LLM to classify the cases more easily
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

		// The logic should now be clean, deterministic, and based solely on the LLM's classification.
		// The LLM classifies the users request by using structured responses
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
				// Fixed bug, better JSON-Parsing
				val syntheticLeave = JSONObject().apply {
					put("changed_settings", JSONArray().apply {
						put(JSONObject().apply {
							put("leave", true)
						})
					})
				}
				lastLlmJsonResponse = syntheticLeave.toString()

				// Ask for confirmation
				speakAndHandleUi("Möchten Sie die Einstellungen wirklich verlassen?")
				StateUpdate(State.SETTINGS_ACTION, lastLlmJsonResponse)
			}
			SettingIntent.NONE -> {
				// Fallback
				val response = "Ich habe das leider nicht verstanden. Sie können die Sprechgeschwindigkeit anpassen, die Stimme ändern oder die Einstellungen verlassen."
				speakAndHandleUi(response)
				StateUpdate(State.SETTINGS_MENU, lastLlmJsonResponse) // Stay in the menu
			}
		}
	}


	// Handling the setting value choice
	suspend fun handleSettingsChoice(final: String): StateUpdate {
		val prompt = "Führe die folgende Aktion aus: '$final'."
		val jsonResponse = generateLlmResponse(prompt, true) ?: return StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)

		lastLlmJsonResponse = jsonResponse
		val confirmationQuestion = jsonParser.createConfirmationQuestion(jsonResponse)

		speakAndHandleUi(confirmationQuestion)
		return StateUpdate(State.SETTINGS_ACTION, jsonResponse)
	}

	// Handling the confirmation for a setting change
	suspend fun handleSettingsAction(final: String): StateUpdate {
		val prompt =
			"Würdest du sagen der Nutzer hat diesen Command bestätigt? Die Antwort des Nutzers war $final. Antworte bitte mit einer JSON-Antwort in approval."
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

	// Generates the LLM response
	private suspend fun generateLlmResponse(prompt: String, structured: Boolean): String? {
		val promptTrimmed = prompt.trim()
		if (promptTrimmed.isEmpty()) {
			Log.w(EyeAIApp.APP_LOG_TAG, "generateLlmResponse: empty prompt - skipping LLM call (structured=$structured)")
			return null
		}

		val promptPreview = if (promptTrimmed.length > 300) promptTrimmed.take(300) + "..." else promptTrimmed
		val start = System.nanoTime()
		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"LLM generate START (structured=$structured) promptPreview='${promptPreview}' at ${System.currentTimeMillis()}"
		)

		return try {
			val result = eyeAIApp.llm!!.generate(promptTrimmed, structured)
			val dur = elapsedMs(start)
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"LLM generate END (structured=$structured) duration=${dur} ms resultPreview='${result.take(300)}'"
			)
			result
		} catch (e: Exception) {
			val dur = elapsedMs(start)
			Log.e(EyeAIApp.APP_LOG_TAG, "LLM generate EXCEPTION after $dur ms", e)
			val errorMsg = if (structured) "Entschuldigung, die strukturierte Anfrage ist fehlgeschlagen."
			else "Entschuldigung, bei der Anfrage ist ein Fehler aufgetreten."
			speakAndHandleUi(errorMsg)
			null
		}
	}

	// Speaks the given text and updates the UI
	private suspend fun speakAndHandleUi(text: String) {
		val toSpeak = text.trim()
		if (toSpeak.isEmpty()) {
			Log.w(EyeAIApp.APP_LOG_TAG, "speakAndHandleUi: empty text, skipping TTS")
			return
		}

		withContext(Dispatchers.Main) {
			llmResponseText?.text = eyeAIApp.getString(R.string.llm_response, toSpeak)
		}
		val ttsEnqueueStart = System.nanoTime()
		Log.d(EyeAIApp.APP_LOG_TAG, "TTS speak() ENQUEUE at ${System.currentTimeMillis()} (textPreview='${toSpeak.take(200)}')")
		textToSpeechInstance.speak(toSpeak)
		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"TTS speak() returned to caller after ${elapsedMs(ttsEnqueueStart)} ms (speak enqueued or returned)"
		)
	}

	// =================================================================================
	//  Using an extra private inner class in order to avoid Json-Parsing in the main functions
	// =================================================================================
	private inner class JsonParser {

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
					else -> RequestedFunction.NONE
				}
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseRequestedFunction", e)
				RequestedFunction.NONE
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
							return if (voice == "1") "Verstanden. Soll die Assistenstimme nun männlich sein?"
							else "Verstanden. Soll die Assistenstimme nun weiblich sein?"
						}
						firstChange.has("leave") -> return "Möchten Sie die Einstellungen wirklich verlassen?"
					}
				}
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in createConfirmationQuestion", e)
			}
			return "Soll ich die angeforderte Änderung durchführen?" // Fallback
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