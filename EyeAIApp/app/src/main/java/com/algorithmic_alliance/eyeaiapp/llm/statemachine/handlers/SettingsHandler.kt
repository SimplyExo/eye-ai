package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import android.content.Context
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.Settings
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationLabel
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModel
import com.algorithmic_alliance.eyeaiapp.llm.LLM
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateUpdate
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.VoskRestartPolicy
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

class SettingsHandler(
	private val textToSpeechInstance: TextToSpeechInstance,
	private val jsonParser: JsonParser,
	private val eyeAIApp: EyeAIApp,
	confirmationModelProvider: () -> ConfirmationModel,
	private val generateLlmResponse: suspend (String, Boolean) -> String?,
	private val speakAndHandleUi: suspend (String) -> Unit
) {
	private val settingsExtractor = GeminiSettingsExtractor(
		jsonParser = jsonParser,
		trace = ::logDecisionTrace,
		generateLlmResponse = generateLlmResponse
	)
	private val settingsConfirmation = LocalSettingsConfirmation(
		confirmationModelProvider = confirmationModelProvider,
		jsonParser = jsonParser,
		trace = ::logDecisionTrace
	)

	private fun logDecisionTrace(message: String) {
		Log.d(EyeAIApp.APP_LOG_TAG, message)
	}

	private fun logConfirmationTransition(
		role: String,
		decision: String,
		action: String,
		nextState: State,
		contextRetained: Boolean,
		modelInvoked: Boolean = true
	) {
		val evaluator = if (modelInvoked) "LOCAL_CONFIRMATION_MODEL" else "STATE_MACHINE_CONTROL"
		Log.i(
			EyeAIApp.APP_LOG_TAG,
			"[DecisionTrace][StateMachine][CONFIRMATION_TRANSITION] " +
				"state=SETTINGS_ACTION role=$role evaluator=$evaluator " +
				"apiCalled=false modelInvoked=$modelInvoked decision=$decision " +
				"action=$action nextState=$nextState contextRetained=$contextRetained"
		)
	}

	suspend fun handleSettingsMenu(
		input: String,
		currentJson: String?,
		onJsonUpdate: (String?) -> Unit
	): StateUpdate {
		// Method when NLP fails
		// NLP logic majorly in StateMachine.kt

		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"[DecisionTrace][Gemini API][CLASSIFY] role=GUIDED_SETTINGS_INTENT input='$input'"
		)
		val intentPrompt = eyeAIApp.llm!!.buildSettingsMenuPrompt(input)
		val jsonResponse = generateLlmResponse(intentPrompt, true)

		if (jsonResponse == null) {
			speakAndHandleUi("LLM-Antwort konnte nicht generiert werden.")
			return StateUpdate(State.SETTINGS_MENU, currentJson)
		}

		val settingIntent = jsonParser.parseSettingIntent(jsonResponse)
		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"[DecisionTrace][Gemini API][RESULT] role=GUIDED_SETTINGS_INTENT " +
				"settingIntent=$settingIntent"
		)

		return when (settingIntent) {
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
		val currentIntent = currentJson?.let { jsonParser.parseSettingIntent(it) } ?: SettingIntent.NONE
		val settings = Settings.load(eyeAIApp)
		val voicePreferences = eyeAIApp.getSharedPreferences("tts_settings", Context.MODE_PRIVATE)
		val snapshot = CurrentSettingsSnapshot(
			speechRate = textToSpeechInstance.speechRate,
			voice = voicePreferences.getInt("tts_voice", 0),
			frequency = settings.depthAudioFrequency,
			bps = settings.depthAudioClickIncidence
		)

		return when (
			val result = settingsExtractor.extract(currentIntent, input, currentJson, snapshot)
		) {
			SettingsExtractionResult.Failed -> {
				speakAndHandleUi("LLM-Antwort konnte nicht generiert werden.")
				StateUpdate(State.SETTINGS_CHOICE, currentJson)
			}

			is SettingsExtractionResult.MissingValue -> {
				speakAndHandleUi(result.targetedQuestion)
				onJsonUpdate(currentJson)
				StateUpdate(State.SETTINGS_CHOICE, currentJson)
			}

			is SettingsExtractionResult.InvalidResponse -> {
				Log.w(
					EyeAIApp.APP_LOG_TAG,
					"[DecisionTrace][SettingsHandler][SETTINGS_CHOICE] " +
						"outcome=INVALID_GEMINI_RESPONSE action=REQUEST_REPHRASE " +
						"nextState=SETTINGS_CHOICE contextRetained=true"
				)
				speakAndHandleUi(result.recoveryQuestion)
				onJsonUpdate(currentJson)
				StateUpdate(State.SETTINGS_CHOICE, currentJson)
			}

			is SettingsExtractionResult.Complete -> {
				speakAndHandleUi(result.confirmationQuestion)
				onJsonUpdate(result.json)
				StateUpdate(State.SETTINGS_ACTION, result.json)
			}
		}
	}

	suspend fun handleSettingsAction(
		input: String,
		currentJson: String?,
		onJsonUpdate: (String?) -> Unit
	): StateUpdate {

		if (currentJson != null && jsonParser.isLeaveRequest(currentJson)) {
			if (ExplicitSettingsFlowAbort.matches(input)) {
				logConfirmationTransition(
					role = "LEAVE_SETTINGS_CONFIRMATION",
					decision = "ABORTED",
					action = "ABORT_SETTINGS_FLOW",
					nextState = State.IDLE,
					contextRetained = false,
					modelInvoked = false
				)
				speakAndHandleUi("Okay, ich habe den Einstellungsdialog abgebrochen.")
				onJsonUpdate(null)
				return StateUpdate(State.IDLE, null)
			}
			return when (settingsConfirmation.evaluate(input, currentJson)) {
				ConfirmationLabel.ACCEPT -> {
					logConfirmationTransition(
						"LEAVE_SETTINGS_CONFIRMATION", "ACCEPT",
						"LEAVE_SETTINGS", State.IDLE, false
					)
					speakAndHandleUi("Die Einstellungen werden verlassen.")
					onJsonUpdate(null)
					StateUpdate(State.IDLE, null)
				}

				ConfirmationLabel.REJECT -> {
					logConfirmationTransition(
						"LEAVE_SETTINGS_CONFIRMATION", "REJECT",
						"STAY_IN_SETTINGS", State.SETTINGS_MENU, false
					)
					speakAndHandleUi("Okay, Sie bleiben in den Einstellungen. Hier sind ihre Funktionen im Einstellungsmenü: Sprachgeschwindigkeit ändern, Stimme ändern, Schläge pro Sekunde ändern, Frequenz anpassen, Einstellungen verlassen.")
					onJsonUpdate(null)
					StateUpdate(State.SETTINGS_MENU, null)
				}

				ConfirmationLabel.UNKNOWN -> {
					logConfirmationTransition(
						"LEAVE_SETTINGS_CONFIRMATION", "UNKNOWN",
						"REQUEST_CLARIFICATION", State.SETTINGS_ACTION, true
					)
					speakAndHandleUi(
						"Ich konnte die Bestätigung nicht eindeutig zuordnen. " +
							"Bitte antworten Sie mit Ja oder Nein."
					)
					StateUpdate(State.SETTINGS_ACTION, currentJson)
				}

				null -> {
					logConfirmationTransition(
						"LEAVE_SETTINGS_CONFIRMATION", "FAILED",
						"KEEP_CONFIRMATION_PENDING", State.SETTINGS_ACTION, true
					)
					speakAndHandleUi("Fehler bei der Verarbeitung.")
					StateUpdate(State.SETTINGS_ACTION, currentJson)
				}
			}
		}

		val settingsFlow = jsonParser.parseSettingsFlow(currentJson)
		return when (
			settingsConfirmation.confirmAndApply(input, currentJson, ::applySettings)
		) {
			SettingsConfirmationResult.APPLIED -> {
				logConfirmationTransition(
					"SETTINGS_CONFIRMATION", "ACCEPT",
					"APPLY_SETTINGS", State.IDLE, false
				)
				Log.i(
					EyeAIApp.APP_LOG_TAG,
					"[DecisionTrace][SettingsHandler][APPLY] outcome=SUCCESS; " +
						"Vosk will require a button press before listening again"
				)
				onJsonUpdate(null)
				StateUpdate(
					newState = State.IDLE,
					newJson = null,
					voskRestartPolicy = VoskRestartPolicy.REQUIRE_MANUAL_RESTART
				)
			}

			SettingsConfirmationResult.REJECTED -> {
				onJsonUpdate(null)
				if (settingsFlow.cancellationDestination() == SettingsCancellationDestination.IDLE) {
					logConfirmationTransition(
						"SETTINGS_CONFIRMATION", "REJECT",
						"CANCEL_SETTING_CHANGE", State.IDLE, false
					)
					speakAndHandleUi("Okay, ich habe die Einstellungsänderung abgebrochen.")
					StateUpdate(State.IDLE, null)
				} else {
					logConfirmationTransition(
						"SETTINGS_CONFIRMATION", "REJECT",
						"RETURN_TO_SETTINGS_MENU", State.SETTINGS_MENU, false
					)
					speakAndHandleUi("Okay, ich habe den Vorgang abgebrochen. Hier sind ihre Funktionen im Einstellungsmenü: Sprachgeschwindigkeit ändern, Stimme ändern, Schläge pro Sekunde ändern, Frequenz anpassen, Einstellungen verlassen.")
					StateUpdate(State.SETTINGS_MENU, null)
				}
			}

			SettingsConfirmationResult.ABORTED -> {
				logConfirmationTransition(
					"SETTINGS_CONFIRMATION", "ABORTED",
					"ABORT_SETTINGS_FLOW", State.IDLE, false, modelInvoked = false
				)
				speakAndHandleUi("Okay, ich habe den Einstellungsdialog abgebrochen.")
				onJsonUpdate(null)
				StateUpdate(State.IDLE, null)
			}

			SettingsConfirmationResult.UNKNOWN -> {
				logConfirmationTransition(
					"SETTINGS_CONFIRMATION", "UNKNOWN",
					"REQUEST_CLARIFICATION", State.SETTINGS_ACTION, true
				)
				speakAndHandleUi(
					"Ich konnte die Bestätigung nicht eindeutig zuordnen. " +
						"Bitte antworten Sie mit Ja oder Nein."
				)
				StateUpdate(State.SETTINGS_ACTION, currentJson)
			}

			SettingsConfirmationResult.FAILED -> {
				logConfirmationTransition(
					"SETTINGS_CONFIRMATION", "FAILED",
					"RETURN_TO_IDLE", State.IDLE, false
				)
				speakAndHandleUi("Fehler bei der Verarbeitung.")
				onJsonUpdate(null)
				StateUpdate(State.IDLE, null)
			}
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
						Log.d(
							EyeAIApp.APP_LOG_TAG,
							"[DecisionTrace][SettingsHandler][APPLY] evaluator=LOCAL " +
								"setting=TTS_SPEED value=$newSpeed"
						)
						speakAndHandleUi("Die Einstellung wurde erfolgreich geändert.")
					}

					setting.has("voice") -> {
						val voice = setting.getInt("voice")
						textToSpeechInstance.setVoice(voice)
						// Save TTS settings
						ttsEditor.putInt("tts_voice", voice)
						Log.d(
							EyeAIApp.APP_LOG_TAG,
							"[DecisionTrace][SettingsHandler][APPLY] evaluator=LOCAL " +
								"setting=VOICE value=$voice"
						)
						speakAndHandleUi("Die Einstellung wurde erfolgreich geändert.")
					}

					setting.has("frequency") -> {
						val frequency = setting.getInt("frequency")
						val clampedFreq = frequency.coerceIn(100, 4000)
						val currentBps = settings.depthAudioClickIncidence
						uniffi.NativeLib.setAudioSettings(clampedFreq.toFloat(), currentBps)

						// Save settings
						settings.depthAudioFrequency = clampedFreq
						settings.save(eyeAIApp)
						Log.d(
							EyeAIApp.APP_LOG_TAG,
							"[DecisionTrace][SettingsHandler][APPLY] evaluator=LOCAL " +
								"setting=FREQUENCY value=${clampedFreq}Hz"
						)
						speakAndHandleUi("Die Audio-Frequenz wurde erfolgreich auf $clampedFreq Hz geändert.")
					}

					setting.has("bps") -> {
						val bps = setting.getInt("bps")
						val clampedBps = bps.coerceIn(1, 10)
						val currentFreq = settings.depthAudioFrequency
						uniffi.NativeLib.setAudioSettings(currentFreq.toFloat(), clampedBps)

						// Save settings
						settings.depthAudioClickIncidence = clampedBps
						settings.save(eyeAIApp)
						Log.d(
							EyeAIApp.APP_LOG_TAG,
							"[DecisionTrace][SettingsHandler][APPLY] evaluator=LOCAL " +
								"setting=BPS value=$clampedBps"
						)
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
