package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import android.content.Context
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.Settings
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

	suspend fun handleSettingsMenu(
		input: String,
		currentJson: String?,
		onJsonUpdate: (String?) -> Unit
	): StateUpdate {
		val intentPrompt = eyeAIApp.llm!!.buildSettingsMenuPrompt(input)
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

			SettingIntent.FREQUENCY -> {
				speakAndHandleUi(LLM.Companion.SNIPPET_FREQUENCY)
				onJsonUpdate(jsonResponse)
				StateUpdate(State.SETTINGS_CHOICE, jsonResponse)
			}

			SettingIntent.BPS -> {
				speakAndHandleUi(LLM.Companion.SNIPPET_BPS)
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
				speakAndHandleUi("Ich habe das leider nicht verstanden. Sie können die Sprechgeschwindigkeit, die Stimme, die Frequenz, die BPS anpassen oder die Einstellungen verlassen.")
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
		val settings = Settings.load(eyeAIApp) // Settings-Instanz laden

		val prompt = when (currentIntent) {
			SettingIntent.TTS_SPEED -> {
				"Die aktuelle Sprechgeschwindigkeit ist ${textToSpeechInstance.speechRate}. Der Nutzer möchte folgendes: '$input'. Passe die Geschwindigkeit entsprechend an und erstelle ein JSON mit 'changed_settings' Array mit 'tts_speed' Feld."
			}

			SettingIntent.VOICE -> {
				"Der Nutzer möchte die Assistentenstimme ändern: '$input'. Wenn der Nutzer 'männlich' oder ähnliches sagt, setze 'voice' auf 0. Wenn 'weiblich' oder ähnliches, setze 'voice' auf 1. Erstelle ein JSON mit 'changed_settings' Array mit 'voice' Feld."
			}

			SettingIntent.FREQUENCY -> {
				"Der Nutzer möchte die Audio-Frequenz ändern: '$input'. Die Frequenz muss zwischen 100 und 4000 Hz liegen. Aktuelle Frequenz ist ${settings.depthAudioFrequency} Hz. Erstelle ein JSON mit 'changed_settings' Array mit 'frequency' Feld."
			}

			SettingIntent.BPS -> {
				"Der Nutzer möchte die BPS (Beats per Second) ändern: '$input'. Die BPS müssen zwischen 1 und 10 liegen. Aktuelle BPS ist ${settings.depthAudioClickIncidence}. Erstelle ein JSON mit 'changed_settings' Array mit 'bps' Feld."
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
			speakAndHandleUi("Okay, ich habe den Vorgang abgebrochen. Hier sind ihre Funktionen im Einstellungsmenü: Sprachgeschwindigkeit ändern, Stimme ändern, Schläge pro Sekunde ändern, Frequenz anpassen, Einstellungen verlassen.")
			onJsonUpdate(null)
			StateUpdate(State.SETTINGS_MENU, null)
		}
	}

	private suspend fun applySettings(jsonString: String): Boolean {
		try {
			val changedSettings = JSONObject(jsonString).getJSONArray("changed_settings")
			val settings = Settings.load(eyeAIApp)

			// SharedPrefs for TTS
			val ttsPrefs = eyeAIApp.getSharedPreferences("tts_settings", Context.MODE_PRIVATE)
			val ttsEditor = ttsPrefs.edit()

			for (i in 0 until changedSettings.length()) {
				val setting = changedSettings.getJSONObject(i)

				when {
					setting.has("tts_speed") -> {
						val newSpeed = setting.getDouble("tts_speed").toFloat()
						textToSpeechInstance.setSpeechRate(newSpeed)

						// Save TTS settings
						ttsEditor.putFloat("tts_speech_rate", newSpeed)

						Log.d(EyeAIApp.APP_LOG_TAG, "TTS-Geschwindigkeit wird auf $newSpeed gesetzt.")
						speakAndHandleUi("Die Einstellung wurde erfolgreich geändert.")
					}

					setting.has("voice") -> {
						val voice = setting.getInt("voice")
						textToSpeechInstance.setVoice(voice)

						// Save TTS settigns
						ttsEditor.putInt("tts_voice", voice)

						Log.d(EyeAIApp.APP_LOG_TAG, "Stimme wird auf $voice gesetzt.")
						speakAndHandleUi("Die Einstellung wurde erfolgreich geändert.")
					}

					setting.has("frequency") -> {
						val frequency = setting.getInt("frequency")
						val clampedFreq = frequency.coerceIn(100, 4000)
						val currentBps = settings.depthAudioClickIncidence

						NativeLib.setAudioSettings(clampedFreq, currentBps)

						// Save settings
						settings.depthAudioFrequency = clampedFreq
						settings.save(eyeAIApp)

						Log.d(EyeAIApp.APP_LOG_TAG, "Audio-Frequenz wird auf $clampedFreq Hz gesetzt.")
						speakAndHandleUi("Die Audio-Frequenz wurde erfolgreich auf $clampedFreq Hz geändert.")
					}

					setting.has("bps") -> {
						val bps = setting.getInt("bps")
						val clampedBps = bps.coerceIn(1, 10)
						val currentFreq = settings.depthAudioFrequency

						NativeLib.setAudioSettings(currentFreq, clampedBps)

						// Save settings
						settings.depthAudioClickIncidence = clampedBps
						settings.save(eyeAIApp)

						Log.d(EyeAIApp.APP_LOG_TAG, "Audio-BPS wird auf $clampedBps gesetzt.")
						speakAndHandleUi("Die BPS wurde erfolgreich auf $clampedBps geändert.")
					}
				}
			}

			// save tts settings
			ttsEditor.apply()

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
