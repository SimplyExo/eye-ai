package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import android.content.Context
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.EyeAIState as State
import com.algorithmic_alliance.eyeaiapp.Settings
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudio
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationLabel
import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModel
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.GenericCancellation
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateUpdate
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.VoskRestartPolicy
import com.algorithmic_alliance.eyeaiapp.settingsparser.CurrentSettingsState
import com.algorithmic_alliance.eyeaiapp.settingsparser.LocalSettingsParser
import com.algorithmic_alliance.eyeaiapp.settingsparser.SettingsCommandExecutor
import com.algorithmic_alliance.eyeaiapp.settingsparser.SpeakerChoice
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONException
import org.json.JSONObject

class SettingsHandler(
	private val textToSpeechInstance: TextToSpeechInstance,
	private val jsonParser: JsonParser,
	private val eyeAIApp: EyeAIApp,
	confirmationModelProvider: () -> ConfirmationModel,
	private val speakAndHandleUi: suspend (String) -> Unit,
	private val localSettingsParserProvider: () -> LocalSettingsParser? = { eyeAIApp.localSettingsParser },
	private val localSettingsCommandExecutor: SettingsCommandExecutor = SettingsCommandExecutor()
) {
	private val localSettingsDialogFlow = LocalSettingsDialogFlow(
		jsonParser = jsonParser,
		commandExecutor = localSettingsCommandExecutor
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

	/** Command extraction and every parameter follow-up use the local frozen parser path. */
	suspend fun handleSettingsChoice(
		input: String,
		currentJson: String?,
		onJsonUpdate: (String?) -> Unit
	): StateUpdate {
		val settings = Settings.load(eyeAIApp)
		val voicePreferences = eyeAIApp.getSharedPreferences("tts_settings", Context.MODE_PRIVATE)
		val currentState = CurrentSettingsState(
			frequency = settings.depthAudioFrequency,
			bps = settings.depthAudioClickIncidence.toDouble(),
			speechSpeed = textToSpeechInstance.speechRate.toDouble(),
			speaker = when (voicePreferences.getInt("tts_voice", -1)) {
				1 -> SpeakerChoice.MALE
				0 -> SpeakerChoice.FEMALE
				else -> SpeakerChoice.UNSPECIFIED
			}
		)
		val result = try {
			withContext(Dispatchers.Default) {
				localSettingsDialogFlow.process(
					input = input,
					currentJson = currentJson,
					currentState = currentState,
					parser = localSettingsParserProvider()
				)
			}
		} catch (error: Throwable) {
			if (error is CancellationException) throw error
			Log.e(
				EyeAIApp.APP_LOG_TAG,
				"[DecisionTrace][SettingsParser][ROUTE] outcome=UNAVAILABLE " +
					"nextEvaluator=NONE role=SETTINGS_PARAMETER_EXTRACTION",
				error
			)
			localSettingsDialogFlow.localRuntimeUnavailable(currentJson)
		}

		return handleLocalSettingsDialogResult(result, onJsonUpdate)
	}

	private suspend fun handleLocalSettingsDialogResult(
		result: LocalSettingsDialogResult,
		onJsonUpdate: (String?) -> Unit
	): StateUpdate = when (result) {
		is LocalSettingsDialogResult.Ready -> {
			val execution = result.execution
			Log.i(
				EyeAIApp.APP_LOG_TAG,
				"[DecisionTrace][SettingsParser][RESULT] execution=LOCAL apiCalled=false " +
					"target=${execution.command.target} operation=${execution.command.operation} " +
					"status=${execution.command.status} action=REQUEST_CONFIRMATION"
			)
			speakAndHandleUi(result.confirmationQuestion)
			onJsonUpdate(result.confirmationJson)
			StateUpdate(State.SETTINGS_ACTION, result.confirmationJson)
		}

		is LocalSettingsDialogResult.FollowUp -> {
			val command = result.command
			Log.i(
				EyeAIApp.APP_LOG_TAG,
				"[DecisionTrace][SettingsParser][RESULT] execution=LOCAL apiCalled=false " +
					"target=${command?.target ?: result.settingIntent} " +
					"operation=${command?.operation} " +
					"status=${result.status ?: result.diagnostic ?: "NEEDS_CLARIFICATION"} " +
					"diagnostic=${result.diagnostic} action=REQUEST_REPHRASE " +
					"nextState=SETTINGS_CHOICE " +
					"contextRetained=${result.retainedContextJson != null}"
			)
			speakAndHandleUi(result.question)
			onJsonUpdate(result.retainedContextJson)
			StateUpdate(State.SETTINGS_CHOICE, result.retainedContextJson, voskRestartPolicy = VoskRestartPolicy.AUTO_RESTART_AFTER_TTS)
		}
	}

	suspend fun handleSettingsAction(
		input: String,
		currentJson: String?,
		onJsonUpdate: (String?) -> Unit
	): StateUpdate {

		if (currentJson != null && jsonParser.isLeaveRequest(currentJson)) {
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

		return when (
			settingsConfirmation.confirmAndApplyWithResult(input, currentJson, ::applySettings)
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
				logConfirmationTransition(
					"SETTINGS_CONFIRMATION", "REJECT",
					"CANCEL_SETTINGS_ACTION", State.IDLE, false
				)
				onJsonUpdate(null)
				speakAndHandleUi(GenericCancellation.RESPONSE)
				StateUpdate(State.IDLE, null, voskRestartPolicy = VoskRestartPolicy.AUTO_RESTART_AFTER_TTS)
			}

			SettingsConfirmationResult.NOT_APPLIED -> {
				logConfirmationTransition(
					"SETTINGS_CONFIRMATION", "ACCEPT",
					"REJECT_UNAVAILABLE_VOICE", State.IDLE, false
				)
				speakAndHandleUi(
					"Die gewünschte Assistentenstimme ist auf diesem Gerät nicht verfügbar. " +
						"Die bisherige Stimme bleibt aktiv."
				)
				onJsonUpdate(null)
				StateUpdate(State.IDLE, null, voskRestartPolicy = VoskRestartPolicy.REQUIRE_MANUAL_RESTART)
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
				StateUpdate(State.SETTINGS_ACTION, currentJson, voskRestartPolicy = VoskRestartPolicy.AUTO_RESTART_AFTER_TTS)
			}

			SettingsConfirmationResult.FAILED -> {
				logConfirmationTransition(
					"SETTINGS_CONFIRMATION", "FAILED",
					"RETURN_TO_IDLE", State.IDLE, false
				)
				speakAndHandleUi("Fehler bei der Verarbeitung.")
				onJsonUpdate(null)
				StateUpdate(State.IDLE, null, voskRestartPolicy = VoskRestartPolicy.AUTO_RESTART_AFTER_TTS)
			}
		}
	}

	private suspend fun applySettings(jsonString: String): SettingsApplyResult {
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
						if (!textToSpeechInstance.setVoice(voice)) {
							Log.w(
								EyeAIApp.APP_LOG_TAG,
								"[DecisionTrace][SettingsHandler][APPLY] outcome=NOT_APPLIED " +
									"setting=VOICE value=$voice"
							)
							return SettingsApplyResult.NOT_APPLIED
						}
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
						SpatialAudio.setAudioSettings(clampedFreq.toFloat(), currentBps)

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
						SpatialAudio.setAudioSettings(currentFreq.toFloat(), clampedBps)

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
			return SettingsApplyResult.APPLIED
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Fehler bei der Verarbeitung der JSON-Aktion.", e)
			return SettingsApplyResult.FAILED
		}
	}

}
