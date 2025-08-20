package com.algorithmic_alliance.eyeaiapp.llm

import android.content.Context
import android.util.Log
import android.widget.TextView
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import com.algorithmic_alliance.eyeaiapp.MainActivity.State


class StateMachine (private val eyeAIApp: EyeAIApp, private val textToSpeechInstance: TextToSpeechInstance, private var lastLlmJsonResponse: String?, private val llmResponseText: TextView?) {




	// Handling of the IDLE state
	suspend fun handleIdle(final: String): StateUpdate {
		val initialResponse = eyeAIApp.llm!!.generate(final, false)

		return when {
			initialResponse.contains("texterkennung", true) -> {
				val prompt = eyeAIApp.llm!!.buildOcrPrompt(eyeAIApp.ocrModel.lastResult)
				val ocrResponse = eyeAIApp.llm!!.generate(prompt, false)
				speakAndHandleUi(ocrResponse)
				StateUpdate(State.IDLE, lastLlmJsonResponse)
			}
			initialResponse.contains("einstellungen", true) -> {
				val settingsResponse = eyeAIApp.llm!!.generate(LLM.SETTINGS_PROMPT, false)
				speakAndHandleUi(settingsResponse)
				StateUpdate(State.SETTINGS_MENU, lastLlmJsonResponse)
			}
			else -> {
				speakAndHandleUi(initialResponse)
				StateUpdate(State.IDLE, lastLlmJsonResponse)
			}
		}
	}

	suspend fun handleSettingsMenu(final: String): StateUpdate {

		// LLM explains options
		val prompt = "Erkläre kurz die Einstellungsmöglichkeit '$final' und frage, wie die Einstellung geändert werden soll je nach Kontext"
		// TODO: Create individual responses for each adaption
		val response = eyeAIApp.llm!!.generate(prompt, false)
		speakAndHandleUi(response)
		return StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
	}

	// LLM executes user command
	suspend fun handleSettingsChoice(final: String): StateUpdate {


		// 1. Send a prompt to the LLM
		val prompt = "Führe die folgende Aktion aus: '$final'."

		val jsonResponse = try {
			eyeAIApp.llm!!.generate(prompt, true) //Generating a structured response
		} catch (e: Exception) {
			// Catching invalid JSONs

			textToSpeechInstance.speak("LLM hat kein valides JSON-Format geliefert!")

			return StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
		}

		// 2. Saving the last JSON-response
		lastLlmJsonResponse = jsonResponse

		// 3. Parsing JSON to create the request
		var confirmationQuestion = "Soll ich die angeforderte Änderung durchführen?" // Fallback
		try {
			val jsonObject = JSONObject(jsonResponse)
			val changedSettings = jsonObject.optJSONArray("changed_settings")
			if (changedSettings != null && changedSettings.length() > 0) {
				val firstChange = changedSettings.getJSONObject(0)
				if (firstChange.has("tts_speed")) {
					val newSpeed = firstChange.getDouble("tts_speed")
					confirmationQuestion = "Verstanden. Soll ich die Sprachgeschwindigkeit auf ${newSpeed} setzen?"
				}
				if (firstChange.has("voice"))
				{

					val voice = firstChange.getString("voice")
					confirmationQuestion = "Verstanden. Soll ich die Assistentenstimme auf ${voice} setzen?"
				}
			}
		} catch (e: Exception) {

			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed", e)
		}

		// 4. Change state to SETTINGS_ACTION, waiting for confirmation

		speakAndHandleUi(confirmationQuestion)
		return StateUpdate(State.SETTINGS_ACTION, jsonResponse)
	}

	// Handling of settings adaption
	suspend fun handleSettingsAction(final: String): StateUpdate {

		val jsonResponse = try {
			eyeAIApp.llm!!.generate("Würdest du sagen der Nutzer hat diesen Command bestätigt? Die Antwort des Nutzers war $final" +
				"Antworte bitte mit einer JSON-Antwort in approval.", true) //Generating a structured response
		} catch (e: Exception) {
			// Catching invalid JSONs

			textToSpeechInstance.speak("LLM hat kein valides JSON-Format geliefert!")

			return StateUpdate(State.IDLE, lastLlmJsonResponse)
		}


		val jsonObject = JSONObject(jsonResponse)
		val changedSettings = jsonObject.getDouble("approval")


		//Checking whether the user confirms his action
		if (changedSettings.toInt() == 1 && lastLlmJsonResponse != null) {
			try {
				// 1. Parsing the JSONObject
				val jsonObject = JSONObject(lastLlmJsonResponse!!)
				val changedSettings = jsonObject.getJSONArray("changed_settings")

				// 2. Changing the settings
				for (i in 0 until changedSettings.length()) {
					val setting = changedSettings.getJSONObject(i)
					if (setting.has("tts_speed")) {
						//Changing speed
						val newSpeed = setting.getDouble("tts_speed").toFloat()
						textToSpeechInstance.setSpeechRate(newSpeed)
						Log.d(EyeAIApp.APP_LOG_TAG, "TTS-Geschwindigkeit wird auf $newSpeed gesetzt.")
					}
					if (setting.has("voice"))
					{

						val voice = setting.getDouble("voice")
						Log.d(EyeAIApp.APP_LOG_TAG, "Stimme wird auf $voice gesetzt.")
						textToSpeechInstance.setVoice(voice)


					}
					if(setting.has("leave")){
					 //TODO: Add leaving
					}

				}

				// 3. Notifying the user that the changes have been applied
				speakAndHandleUi("Die Einstellung wurde erfolgreich geändert.")

			} catch (e: Exception) {
				Log.e(EyeAIApp.APP_LOG_TAG, "Fehler bei der Verarbeitung der JSON-Aktion.", e)
				speakAndHandleUi("Entschuldigung, beim Anwenden der Einstellung ist ein Fehler aufgetreten.")
			}
		} else {
			// Managing an exit
			speakAndHandleUi("Okay, ich habe den Vorgang abgebrochen.")
		}

		// 4. Clearing up
		lastLlmJsonResponse = null

		return StateUpdate(State.IDLE,null)
	}

	suspend fun speakAndHandleUi(text: String) {
		// UI-Update using the main-thread
		withContext(Dispatchers.Main) {
			llmResponseText?.text = eyeAIApp.getString(R.string.llm_response, text)
		}
		// TTS (using the worker-thread)
		textToSpeechInstance.speak(text)
	}


}