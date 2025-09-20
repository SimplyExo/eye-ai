package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.llm.LLM
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateUpdate
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

class SettingsHandler(
	private val textToSpeechInstance: TextToSpeechInstance,
	private val jsonParser: JsonParser,
	private val eyeAIApp: EyeAIApp,
	private val generateLlmResponse: suspend (String, Boolean) -> String?,
	private val speakAndHandleUi: suspend (String) -> Unit
) {
	companion object {
		private const val SETTINGS_MENU_PROMPT_TEMPLATE = """Der Nutzer ist im Einstellungsmenü und sagt: '$1'.

Klassifiziere die Absicht des Nutzers in eine der folgenden Kategorien und gib sie im Feld 'setting_intent' zurück:

- 'tts_speed': Wenn der Nutzer die Sprechgeschwindigkeit ändern will (z.B. "schneller sprechen").
- 'voice': Wenn der Nutzer die Stimme des Assistenten ändern will (z.B. "Stimme ändern", "andere Stimme", "Assistentenagenten anpassen").
- 'leave': Wenn der Nutzer die Einstellungen verlassen will.
- 'none': Wenn keine der obigen Absichten klar erkennbar ist.

Antworte NUR mit dem JSON-Objekt.

Beispiel für die Eingabe "ich will eine andere Stimme": {"setting_intent": "voice"}
Beispiel für die Eingabe "verlassen": {"setting_intent": "leave"}
"""
	}

	suspend fun handleSettingsMenu(
		input: String,
		currentJson: String?,
		onJsonUpdate: (String?) -> Unit
	): StateUpdate {
		val intentPrompt = SETTINGS_MENU_PROMPT_TEMPLATE.replace("$1", input)
		val jsonResponse = generateLlmResponse(intentPrompt, true)

		if (jsonResponse == null) {
			speakAndHandleUi("LLM-Antwort konnte nicht generiert werden.")
			return StateUpdate(State.SETTINGS_MENU, currentJson)
		}

		return when (jsonParser.parseSettingIntent(jsonResponse)) {
			SettingIntent.TTS_SPEED -> {
				speakAndHandleUi(LLM.Companion.SNIPPET_TTS_SPEED)
				onJsonUpdate(jsonResponse)
				StateUpdate(State.SETTINGS_CHOICE, jsonResponse)
			}
			SettingIntent.VOICE -> {
				speakAndHandleUi(LLM.Companion.SNIPPET_VOICE)
				onJsonUpdate(jsonResponse)
				StateUpdate(State.SETTINGS_CHOICE, jsonResponse)
			}
			SettingIntent.LEAVE -> {
				val syntheticLeave = createLeaveSettingsJson()
				speakAndHandleUi("Möchten Sie die Einstellungen wirklich verlassen?")
				onJsonUpdate(syntheticLeave)
				StateUpdate(State.SETTINGS_ACTION, syntheticLeave)
			}
			SettingIntent.NONE -> {
				speakAndHandleUi("Ich habe das leider nicht verstanden. Sie können die Sprechgeschwindigkeit anpassen, die Stimme ändern oder die Einstellungen verlassen.")
				StateUpdate(State.SETTINGS_MENU, currentJson)
			}
		}
	}

	suspend fun handleSettingsChoice(
		input: String,
		currentJson: String?,
		onJsonUpdate: (String?) -> Unit
	): StateUpdate {
		val currentIntent = currentJson?.let { jsonParser.parseSettingIntent(it) }

		val prompt = when (currentIntent) {
			SettingIntent.TTS_SPEED -> {
				"Die aktuelle Sprechgeschwindigkeit ist ${textToSpeechInstance.speechRate}. Der Nutzer möchte folgendes: '$input'. Passe die Geschwindigkeit entsprechend an und erstelle ein JSON mit 'changed_settings' Array mit 'tts_speed' Feld."
			}
			SettingIntent.VOICE -> {
				"Der Nutzer möchte die Assistentenstimme ändern: '$input'. Wenn der Nutzer 'männlich' oder ähnliches sagt, setze 'voice' auf 0. Wenn 'weiblich' oder ähnliches, setze 'voice' auf 1. Erstelle ein JSON mit 'changed_settings' Array mit 'voice' Feld."
			}
			else -> {
				"Führe die folgende Aktion aus: '$input'."
			}
		}

		val jsonResponse = generateLlmResponse(prompt, true)
		if (jsonResponse == null) {
			speakAndHandleUi("LLM-Antwort konnte nicht generiert werden.")
			return StateUpdate(State.SETTINGS_CHOICE, currentJson)
		}

		val confirmationQuestion = jsonParser.createConfirmationQuestion(jsonResponse)
		speakAndHandleUi(confirmationQuestion)
		onJsonUpdate(jsonResponse)
		return StateUpdate(State.SETTINGS_ACTION, jsonResponse)
	}

	suspend fun handleSettingsAction(
		input: String,
		currentJson: String?,
		onJsonUpdate: (String?) -> Unit
	): StateUpdate {
		val prompt = "Würdest du sagen der Nutzer hat diesen Command bestätigt? Die Antwort des Nutzers war $input. Antworte bitte mit einer JSON-Antwort in approval."
		val jsonResponse = generateLlmResponse(prompt, true)

		if (jsonResponse == null) {
			speakAndHandleUi("Fehler bei der Verarbeitung.")
			onJsonUpdate(null)
			return StateUpdate(State.IDLE, null)
		}

		return if (jsonParser.isApproved(jsonResponse) && currentJson != null) {
			// Adapt settings and save JSOn
			applySettings(currentJson)
			onJsonUpdate(null)
			StateUpdate(State.IDLE, null) // No further message needed.
		} else {
			speakAndHandleUi("Okay, ich habe den Vorgang abgebrochen. Hier sind ihre Funktionen im Einstellungsmenü: Sprachgeschwindigkeit ändern, Stimme ändern, Einstellungen verlassen.")
			onJsonUpdate(null)
			StateUpdate(State.SETTINGS_MENU, null)
		}
	}

	private suspend fun applySettings(jsonString: String): Boolean {
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

	private fun createLeaveSettingsJson(): String {
		return JSONObject().apply {
			put("changed_settings", JSONArray().apply {
				put(JSONObject().apply {
					put("leave", true)
				})
			})
		}.toString()
	}
}
