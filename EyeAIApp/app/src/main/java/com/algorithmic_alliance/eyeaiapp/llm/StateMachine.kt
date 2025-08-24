package com.algorithmic_alliance.eyeaiapp.llm

import android.util.Log
import android.widget.TextView
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONException
import org.json.JSONObject

class StateMachine(
	private val eyeAIApp: EyeAIApp,
	private val textToSpeechInstance: TextToSpeechInstance,
	private var lastLlmJsonResponse: String?,
	private val llmResponseText: TextView?
) {

	//Enum class for requested functions
	enum class RequestedFunction {
		TEXT_RECOGNITION, SETTINGS, NONE
	}

	//private parser
	private val jsonParser = JsonParser()

	fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

	//handling Idle
	suspend fun handleIdle(final: String): StateUpdate {
		val jsonResponse = generateLlmResponse(final, true) ?: return StateUpdate(State.IDLE, null)

		return when (jsonParser.parseRequestedFunction(jsonResponse)) {
			RequestedFunction.TEXT_RECOGNITION -> {
				val prompt = eyeAIApp.llm!!.buildOcrPrompt(eyeAIApp.ocrModel.lastResult)
				val ocrResponse = generateLlmResponse(prompt, false) ?: ""
				speakAndHandleUi(ocrResponse)
				StateUpdate(State.IDLE, null)
			}
			RequestedFunction.SETTINGS -> {
				//Avoiding delays by using snippets to answer as generating an individual response comes with loosing time and is less specific.
				//We still rely on the structured detection above (jsonResponse) to decide that we are in settings.
				//Speak the hardcoded settings menu text and set state to SETTINGS_MENU
				speakAndHandleUi(LLM.SNIPPET_SETTINGS)
				//store last structured json so later confirmations can use it if needed
				lastLlmJsonResponse = jsonResponse
				StateUpdate(State.SETTINGS_MENU, lastLlmJsonResponse)
			}
			RequestedFunction.NONE -> {
				//fallback if no function was named, handling a regular question
				val fallbackResponse = generateLlmResponse(final, false) ?: jsonResponse
				speakAndHandleUi(fallbackResponse)
				StateUpdate(State.IDLE, null)
			}
		}
	}

	//handling the user choice while in SETTINGS_MENU
	suspend fun handleSettingsMenu(final: String): StateUpdate {
		val intentPrompt =
			"Der Nutzer befindet sich im Einstellungsmenü und sagt: '$final'. Prüfe, ob der Nutzer die Einstellungen verlassen möchte oder welche Einstellung er meint (tts_speed / voice / leave)."
		val jsonResponse = generateLlmResponse(intentPrompt, true)

		if (jsonResponse != null && jsonParser.wantsToLeave(jsonResponse)) { //user wants to leave
			speakAndHandleUi("Möchten Sie die Einstellungen wirklich verlassen?")
			// keep the structured response so later steps can act on it
			lastLlmJsonResponse = jsonResponse
			return StateUpdate(State.SETTINGS_ACTION, jsonResponse)
		} else {
			try {
				if (jsonResponse != null) {
					val jsonObj = JSONObject(jsonResponse)
					val changedSettings = jsonObj.optJSONArray("changed_settings")
					if (changedSettings != null && changedSettings.length() > 0) {
						val firstChange = changedSettings.getJSONObject(0)
						//handling a request to change the tts speed
						if (firstChange.has("tts_speed")) {
							lastLlmJsonResponse = jsonResponse
							speakAndHandleUi(LLM.SNIPPET_TTS_SPEED) //using a snippet in order to avoid delays
							return StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
						}
						//handling a request to change the voice
						if (firstChange.has("voice")) {
							lastLlmJsonResponse = jsonResponse
							speakAndHandleUi(LLM.SNIPPET_VOICE) //using a snippet in order to avoid delays
							return StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
						}
						if (firstChange.has("leave")) {
							lastLlmJsonResponse = jsonResponse
							speakAndHandleUi("Möchten Sie die Einstellungen wirklich verlassen?")
							return StateUpdate(State.SETTINGS_ACTION, lastLlmJsonResponse)
						}
					}
				}
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in handleSettingsMenu", e)
			}

			// Fallback: falls keine konkrete changed_settings erkannt wurde, hole eine normale Erklärung per LLM (wie bisher)
			val explanationPrompt =
				"Erkläre kurz die Einstellungsmöglichkeit '$final' und frage, wie die Einstellung geändert werden soll je nach Kontext."
			val response = generateLlmResponse(explanationPrompt, false) ?: "Ich habe das leider nicht verstanden."
			speakAndHandleUi(response)
			return StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
		}
	}


	suspend fun handleSettingsChoice(final: String): StateUpdate {
		val prompt = "Führe die folgende Aktion aus: '$final'."
		val jsonResponse = generateLlmResponse(prompt, true) ?: return StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)

		lastLlmJsonResponse = jsonResponse
		val confirmationQuestion = jsonParser.createConfirmationQuestion(jsonResponse)

		speakAndHandleUi(confirmationQuestion)
		return StateUpdate(State.SETTINGS_ACTION, jsonResponse)
	}

	//handling the requests
	suspend fun handleSettingsAction(final: String): StateUpdate {
		val prompt =
			"Würdest du sagen der Nutzer hat diesen Command bestätigt? Die Antwort des Nutzers war $final. Antworte bitte mit einer JSON-Antwort in approval."
		val jsonResponse = generateLlmResponse(prompt, true) ?: return StateUpdate(State.IDLE, lastLlmJsonResponse)

		if (jsonParser.isApproved(jsonResponse) && lastLlmJsonResponse != null) { //checking for user approval
			val success = jsonParser.applySettings(lastLlmJsonResponse!!)
			if (!success) {
				speakAndHandleUi("Entschuldigung, beim Anwenden der Einstellung ist ein Fehler aufgetreten.")
			}
		} else {
			speakAndHandleUi("Okay, ich habe den Vorgang abgebrochen.")
		}
		return StateUpdate(State.IDLE, null)
	}


	//generate the LlmResponse
	private suspend fun generateLlmResponse(prompt: String, structured: Boolean): String? {
		val promptPreview = if (prompt.length > 300) prompt.take(300) + "..." else prompt
		val start = System.nanoTime()
		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"LLM generate START (structured=$structured) promptPreview='${promptPreview}' at ${System.currentTimeMillis()}"
		)

		return try {
			val result = eyeAIApp.llm!!.generate(prompt, structured)
			val dur = elapsedMs(start)
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"LLM generate END (structured=$structured) duration=${dur} ms resultPreview='${result.take(300)}'"
			)
			result
		} catch (e: Exception) {
			val dur = elapsedMs(start)
			Log.e(EyeAIApp.APP_LOG_TAG, "LLM generate EXCEPTION after $dur ms", e)
			if (structured) {
				speakAndHandleUi("Entschuldigung, die strukturierte Anfrage ist fehlgeschlagen.")
			} else {
				speakAndHandleUi("Entschuldigung, bei der Anfrage ist ein Fehler aufgetreten.")
			}
			null
		}
	}

	private suspend fun speakAndHandleUi(text: String) {
		withContext(Dispatchers.Main) {
			llmResponseText?.text = eyeAIApp.getString(R.string.llm_response, text)
		}
		val ttsEnqueueStart = System.nanoTime()
		Log.d(EyeAIApp.APP_LOG_TAG, "TTS speak() ENQUEUE at ${System.currentTimeMillis()} (textPreview='${text.take(200)}')")
		textToSpeechInstance.speak(text) // <-- sicherstellen, dass speak nicht blockiert
		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"TTS speak() returned to caller after ${elapsedMs(ttsEnqueueStart)} ms (speak enqueued or returned)"
		)
	}


	// =================================================================================
	//  Using an extra private inner class in order to avoid Json-Parsing in the main functions
	// =================================================================================

	private inner class JsonParser {
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

		fun wantsToLeave(jsonString: String): Boolean {
			return try {
				val changedSettings = JSONObject(jsonString).optJSONArray("changed_settings")
				if (changedSettings != null && changedSettings.length() > 0) {
					changedSettings.getJSONObject(0).optBoolean("leave", false)
				} else {
					false
				}
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in wantsToLeave", e)
				false
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
							return if (voice.equals("1")) "Verstanden. Soll die Assistenstimme nun männlich sein?"
							else "Verstanden. Soll die Assistenstimme nun weiblich sein?"
						}
						firstChange.has("leave") -> return "Möchten Sie die Einstellungen wirklich verlassen?"
					}
				}
			} catch (e: JSONException) {
				Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in createConfirmationQuestion", e)
			}
			return "Soll ich die angeforderte Änderung durchführen?" //fallback
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


