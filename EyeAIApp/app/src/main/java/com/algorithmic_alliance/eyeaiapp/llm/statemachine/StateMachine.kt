package com.algorithmic_alliance.eyeaiapp.llm.statemachine

import android.util.Log
import android.widget.TextView
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.JsonParser
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.SpeechOutputHandler
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.ContextSwitchConfirmationResult
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.ContextSwitchConfirmation
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.PendingExternalIntent
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.PendingExternalIntentCodec
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.PendingExternalIntentPresentation
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects.ObjectDetectionHandler
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.SettingIntent
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.missingOperationQuestion
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.SettingsFlow
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.SettingsHandler
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.SettingsIntentRoute
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.SettingsIntentRouter
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.SettingsMenuIntentRoute
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.SettingsMenuIntentRouter
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import kotlinx.coroutines.delay
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects.ObjectPositionClassifier
import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import com.algorithmic_alliance.eyeaiapp.nlp.IntentResult
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModel
import com.algorithmic_alliance.eyeaiapp.nlp.OCRToText
import com.algorithmic_alliance.eyeaiapp.AIModelData
import java.util.Locale

class StateMachine(
    private val eyeAIApp: EyeAIApp,
    private val textToSpeechInstance: TextToSpeechInstance,
	private var lastDialogContext: String?,
	private val setSpeechResponseText: (String) -> Unit,
	private val cameraFrameAnalyzer: CameraFrameAnalyzer? = null
) {

	private val speechOutputHandler = SpeechOutputHandler(
		textToSpeechInstance,
		updateResponseText = setSpeechResponseText
	)
    private val jsonParser = JsonParser()

    // SettingsHandler
    private val settingsHandler = SettingsHandler(
        textToSpeechInstance,
        jsonParser,
        eyeAIApp,
        { eyeAIApp.confirmationModel },
        speechOutputHandler::speakAndHandleUi
    )
    private val contextSwitchConfirmation = ContextSwitchConfirmation(
        confirmationModelProvider = { eyeAIApp.confirmationModel },
        trace = { message -> Log.d(EyeAIApp.APP_LOG_TAG, message) }
    )

    private val objectPositionClassifier = ObjectPositionClassifier()
    private val ocrToText = OCRToText()

    // NLP model integration
    private val nlpModel: NLPModel? = eyeAIApp.nlpModel

    /** Complete, unfiltered NLP V2 result for future state-aware consumers. */
    var lastIntentResult: IntentResult? = null
        private set

    // Keep the classifier acceptance threshold unchanged; rejected results are local unresolved commands.
    private val nlpConfidenceThreshold = 0.6f

    suspend fun handleIdle(final: String): StateUpdate {
        val intentResult = classifyIntentWithNLP(final)
        val confidence = intentResult?.confidence ?: 0f
        val nlpIntent = intentResult?.intent?.takeIf { confidence >= nlpConfidenceThreshold }

        if (nlpIntent == null) {
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][ROUTE] classifier=NLP_V2 accepted=false " +
                    "confidence=${formatPercentage(confidence)} threshold=${formatPercentage(nlpConfidenceThreshold)} " +
                    "nextEvaluator=LOCAL_UNRESOLVED role=UNRESOLVED_COMMAND"
            )
            return handleUnresolvedCommand(classifierLabel = intentResult?.intent)
        }

        val confidentIntentResult = requireNotNull(intentResult)
        return when (val settingsRoute = SettingsIntentRouter.route(confidentIntentResult)) {
            SettingsIntentRoute.GuidedMenu -> {
                logNlpRoute(nlpIntent, "LOCAL_STATE_MACHINE", "OPEN_GUIDED_SETTINGS_MENU")
                openGuidedSettingsMenu()
            }

            is SettingsIntentRoute.Direct -> {
                logNlpRoute(
                    nlpIntent,
                    "LOCAL_SETTINGS_PARSER",
                    "DIRECT_SETTINGS_PARSE"
                )
                handleDirectSettingsRequest(settingsRoute)
            }

            SettingsIntentRoute.NotSettings -> when (nlpIntent) {
                Intent.TEXT_RECOGNITION -> {
                    logNlpRoute(nlpIntent, "LOCAL_OCR_PIPELINE", "TEXT_RECOGNITION")
                    handleTextRecognitionDirectly()
                }

                Intent.OBJECT_DETECTION -> {
                    logNlpRoute(nlpIntent, "LOCAL_OBJECT_DETECTION", "OBJECT_QUERY_EVALUATION")
                    handleObjectDetectionRequest(final)
                }

                Intent.MEASURE_DISTANCE -> {
                    logNlpRoute(nlpIntent, "LOCAL_UNRESOLVED", "UNSUPPORTED_DISTANCE_REQUEST")
                    handleUnresolvedCommand(classifierLabel = Intent.MEASURE_DISTANCE)
                }

				Intent.REDIRECT_TO_LLM -> {
					logNlpRoute(nlpIntent, "LOCAL_UNRESOLVED", "UNRESOLVED_COMMAND")
					handleUnresolvedCommand(classifierLabel = Intent.REDIRECT_TO_LLM)
				}

				Intent.ABORT -> {
					logNlpRoute(nlpIntent, "LOCAL_STATE_MACHINE", "GENERIC_CANCEL")
					handleCancellation()
                }

                else -> {
                    logNlpRoute(nlpIntent, "LOCAL_UNRESOLVED", "UNSUPPORTED_INTENT")
                    handleUnresolvedCommand(classifierLabel = nlpIntent)
                }
            }
        }
    }

    private suspend fun handleUnresolvedCommand(
        nextState: State = State.IDLE,
        retainedContext: String? = null,
        classifierLabel: Intent? = null
    ): StateUpdate {
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][UNRESOLVED] " +
                "classifierLabel=${classifierLabel?.name ?: "NONE"} semantic=UNRESOLVED_COMMAND " +
                "externalRequest=false"
        )
        speechOutputHandler.speakAndHandleUi(LocalInteractionMessages.UNRESOLVED_COMMAND)
        return StateUpdate(nextState, retainedContext, voskRestartPolicy = VoskRestartPolicy.AUTO_RESTART_AFTER_TTS)
    }

    private fun classifyIntentWithNLP(input: String): IntentResult? {
        lastIntentResult = null
        val activeModel = nlpModel
        if (activeModel == null) {
            Log.w(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][NLP V2][CLASSIFY] outcome=UNAVAILABLE input='$input'"
            )
            return null
        }

        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][NLP V2][CLASSIFY] model=${activeModel.info.id} input='$input'"
        )
        return try {
            activeModel.classify(input).also { result ->
                lastIntentResult = result
                val probabilities = Intent.CLASS_ORDER.indices.joinToString(
                    prefix = "[",
                    postfix = "]"
                ) { index ->
                    "${Intent.CLASS_ORDER[index].name}=" +
                        String.format(Locale.US, "%.4f", result.probabilities[index])
                }
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][NLP V2][RESULT] model=${activeModel.info.id} " +
                        "top1=${result.intent} confidence=${formatPercentage(result.confidence)} " +
                        "probabilities=$probabilities originalText='${result.originalText}'"
                )
            }
        } catch (e: Exception) {
            Log.e(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][NLP V2][RESULT] model=${activeModel.info.id} outcome=FAILED",
                e
            )
            lastIntentResult = null
            null
        }
    }

    private fun logNlpRoute(intent: Intent, nextEvaluator: String, role: String) {
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][ROUTE] classifier=NLP_V2 accepted=true " +
                "intent=$intent nextEvaluator=$nextEvaluator role=$role"
        )
    }

	private fun formatPercentage(probability: Float): String =
		String.format(Locale.US, "%.2f%%", probability * 100f)

	suspend fun handleCancellation(): StateUpdate {
		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"[DecisionTrace][StateMachine][CANCEL] evaluator=GENERIC_CANCELLATION " +
				"outcome=RETURN_TO_IDLE pendingContext=cleared"
		)
		lastDialogContext = null
		speechOutputHandler.speakAndHandleUi(GenericCancellation.RESPONSE)
		return StateUpdate(State.IDLE, null)
	}

    private suspend fun handleTextRecognitionDirectly(): StateUpdate {
        val ocrSuccess = cameraFrameAnalyzer?.runOcrAnalysis() ?: false

        if (!ocrSuccess) {
            Log.d(EyeAIApp.APP_LOG_TAG, "OCR analysis failed")
            speechOutputHandler.speakAndHandleUi("Entschuldigung, die Texterkennung konnte nicht durchgeführt werden.")
            return StateUpdate(State.IDLE, null)
        }

        delay(200) // Wait for OCR result to be available

        // Get OCR boxes and convert Array to List
        val ocrBoxes = AIModelData.ocrBoxes.get()

        if (ocrBoxes.isNullOrEmpty()) {
            Log.d(EyeAIApp.APP_LOG_TAG, "No OCR boxes available after OCR analysis")
            speechOutputHandler.speakAndHandleUi("Entschuldigung, es wurde kein Text erkannt.")
            return StateUpdate(State.IDLE, null)
        }

        // Convert Array to List for OCRToText
        val ocrBoxesList = ocrBoxes.toList()

        // Use OCRToText to generate description
        val readableText = ocrToText.generateReadableText(ocrBoxesList)

        if (readableText.isEmpty()) {
            Log.d(EyeAIApp.APP_LOG_TAG, "OCRToText generated empty text")
            speechOutputHandler.speakAndHandleUi("Entschuldigung, ich konnte keinen sinnvollen Text erkennen.")
            return StateUpdate(State.IDLE, null)
        }

        Log.d(EyeAIApp.APP_LOG_TAG, "OCRToText generated: $readableText")
        speechOutputHandler.speakAndHandleUi(readableText)

        return StateUpdate(State.IDLE, null)
    }

    private suspend fun openGuidedSettingsMenu(): StateUpdate {
        speechOutputHandler.speakAndHandleUi(LocalInteractionMessages.SETTINGS_MENU)
        return StateUpdate(State.SETTINGS_MENU, null)
    }

    private suspend fun handleDirectSettingsRequest(
        route: SettingsIntentRoute.Direct
    ): StateUpdate {
        val intentResult = route.intentResult
        val settingsContext = jsonParser.createSettingsContext(
            settingIntent = route.settingIntent,
            flow = SettingsFlow.DIRECT,
            originalText = intentResult.originalText
        )
        lastDialogContext = settingsContext

        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][HANDOFF] classifier=NLP_V2 " +
                "intent=${intentResult.intent} evaluator=LOCAL_SETTINGS_PARSER " +
                "role=SETTINGS_PARAMETER_EXTRACTION " +
                "settingIntent=${route.settingIntent} " +
                "originalText='${intentResult.originalText}'"
        )

        return settingsHandler.handleSettingsChoice(
            input = intentResult.originalText,
            currentJson = settingsContext
        ) { newJson ->
            lastDialogContext = newJson
        }
    }

    private suspend fun handleObjectDetectionRequest(userInput: String): StateUpdate {
        val germanObjectQuery = userInput.trim()
        Log.d(EyeAIApp.APP_LOG_TAG, "German object query from local NLP input: '$germanObjectQuery'")

        if (germanObjectQuery.isBlank()) {
            Log.w(EyeAIApp.APP_LOG_TAG, "Object query is blank")
            val availableGermanObjects = ObjectDetectionHandler.getGermanObjectLabels()
            Log.d(EyeAIApp.APP_LOG_TAG, "Available German objects: $availableGermanObjects")

            if (availableGermanObjects.isNotEmpty()) {
                val objectList = availableGermanObjects.take(5).joinToString(", ")
                speechOutputHandler.speakAndHandleUi(
                    "Bitte nennen Sie ein spezifisches Objekt, nach dem Sie suchen möchten. " +
                        "Verfügbare Objekte: $objectList"
                )
            } else {
                speechOutputHandler.speakAndHandleUi("Entschuldigung, ich konnte gerade keine Objekte erkennen.")
            }
            return StateUpdate(State.IDLE, null)
        }

        Log.d(EyeAIApp.APP_LOG_TAG, "Calling ObjectDetectionHandler.handleGermanObjectQuery with: '$germanObjectQuery'")

        // Looking for the german object instantly
        when (val result = ObjectDetectionHandler.handleGermanObjectQuery(germanObjectQuery)) {
            is ObjectDetectionHandler.ObjectDetectionResult.ObjectFound -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "Object FOUND: ${result.obj}")
                val obj = result.obj
                val objectData = ObjectPositionClassifier.ObjectData(
                    label = obj.label,
                    height = obj.height,
                    width = obj.width,
                    x = obj.x,
                    y = obj.y,
                    distance = obj.distance
                )

                val description = objectPositionClassifier.generatePositionDescription(objectData)
                Log.d(EyeAIApp.APP_LOG_TAG, "Generated position description: $description")
                speechOutputHandler.speakAndHandleUi(description)
            }

            is ObjectDetectionHandler.ObjectDetectionResult.ObjectNotFound -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "Object NOT FOUND. Available: ${result.availableObjects}")
                speechOutputHandler.speakAndHandleUi(
                    "Entschuldigung, das Objekt '$germanObjectQuery' konnte ich nicht finden. " +
                        "Ich sehe aber folgende Objekte: ${result.availableObjects.joinToString(", ")}."
                )
            }

            is ObjectDetectionHandler.ObjectDetectionResult.NoObjectsFound -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "No objects found")
                speechOutputHandler.speakAndHandleUi("Entschuldigung, ich konnte gerade keine Objekte erkennen.")
            }

            is ObjectDetectionHandler.ObjectDetectionResult.DepthDataUnavailable -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "Depth data unavailable")
                speechOutputHandler.speakAndHandleUi("Entschuldigung, die Tiefenerkennung ist derzeit nicht verfügbar.")
            }

            is ObjectDetectionHandler.ObjectDetectionResult.DepthDataInvalid -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "Depth data invalid")
                speechOutputHandler.speakAndHandleUi("Entschuldigung, die Tiefendaten haben eine unerwartete Größe.")
            }

            is ObjectDetectionHandler.ObjectDetectionResult.NoKnownObjectsFound -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "No known objects found")
                speechOutputHandler.speakAndHandleUi("Ich konnte leider keine bekannten Objekte erkennen.")
            }

            is ObjectDetectionHandler.ObjectDetectionResult.NoQueryProvided -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "No query provided")
                speechOutputHandler.speakAndHandleUi("Bitte nennen Sie ein spezifisches Objekt.")
            }
        }

        return StateUpdate(State.IDLE, null)
    }


    suspend fun handleSettingsMenu(final: String): StateUpdate {
        val intentResult = classifyIntentWithNLP(final)
        if (intentResult == null) {
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][SETTINGS_ROUTE] classifier=NLP_V2 " +
                    "outcome=UNAVAILABLE nextEvaluator=LOCAL_UNRESOLVED role=UNRESOLVED_COMMAND"
            )
            return handleUnresolvedCommand(
                nextState = State.SETTINGS_MENU,
                retainedContext = lastDialogContext,
                classifierLabel = null
            )
        }

        val evidence = SettingsMenuIntentRouter.evidenceFrom(intentResult)
        val route = SettingsMenuIntentRouter.route(evidence, nlpConfidenceThreshold)
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][SETTINGS_ROUTE] classifier=NLP_V2 " +
                "top1=${evidence.topIntent} top1Confidence=${formatPercentage(evidence.topConfidence)} " +
                "bestSettingsIntent=${evidence.bestSettingsIntent} " +
                "bestSettingsConfidence=${formatPercentage(evidence.bestSettingsConfidence)} " +
                "threshold=${formatPercentage(nlpConfidenceThreshold)} route=${route::class.simpleName} " +
                "probabilitiesPreserved=true renormalized=false"
        )

        return when (route) {
            is SettingsMenuIntentRoute.LocalSetting ->
                handleLocalSettingsMenuIntent(route.intent)

            is SettingsMenuIntentRoute.ExternalIntent ->
                requestExternalIntentContextSwitch(intentResult)

            SettingsMenuIntentRoute.Abort ->
                handleCancellation()

            SettingsMenuIntentRoute.AlreadyInSettings ->
                remindUserSettingsAreAlreadyOpen()

            SettingsMenuIntentRoute.Unresolved ->
                handleUnresolvedCommand(
                    nextState = State.SETTINGS_MENU,
                    retainedContext = lastDialogContext,
                    classifierLabel = intentResult.intent
                )
        }
    }

    private suspend fun handleLocalSettingsMenuIntent(intent: Intent): StateUpdate {
        val settingIntent = when (intent) {
            Intent.CHANGE_SPEECH_SPEED -> SettingIntent.TTS_SPEED
            Intent.CHANGE_SPEAKER -> SettingIntent.VOICE
            Intent.SET_FREQUENCY -> SettingIntent.FREQUENCY
            Intent.SET_BPS -> SettingIntent.BPS
            else -> error("Intent $intent is not a concrete settings intent")
        }
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][SETTINGS_ROUTE] classifier=NLP_V2 " +
                "intent=$intent action=LOCAL_SETTINGS_SELECTION nextState=SETTINGS_CHOICE"
        )
        speechOutputHandler.speakAndHandleUi(settingIntent.missingOperationQuestion())
        lastDialogContext = jsonParser.createSettingsContext(settingIntent)
        return StateUpdate(State.SETTINGS_CHOICE, lastDialogContext)
    }

    private suspend fun requestExternalIntentContextSwitch(
        intentResult: IntentResult
    ): StateUpdate {
        val pendingIntent = PendingExternalIntent(intentResult)
        val pendingContext = PendingExternalIntentCodec.encode(pendingIntent)
        lastDialogContext = pendingContext
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] action=STORE " +
                "intent=${intentResult.intent} confidence=${formatPercentage(intentResult.confidence)} " +
                "originalText='${intentResult.originalText}' probabilities=${intentResult.probabilities.size}"
        )
        speechOutputHandler.speakAndHandleUi(
            PendingExternalIntentPresentation.confirmationQuestion(intentResult.intent)
        )
        return StateUpdate(State.SETTINGS_EXTERNAL_CONFIRMATION, pendingContext)
    }

    private suspend fun remindUserSettingsAreAlreadyOpen(): StateUpdate {
        lastDialogContext = null
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][SETTINGS_ROUTE] classifier=NLP_V2 " +
                "intent=OPEN_SETTINGS action=REPLAY_EXISTING_MENU nestedFlowCreated=false"
        )
        speechOutputHandler.speakAndHandleUi(
            "Sie befinden sich bereits in den Einstellungen. ${LocalInteractionMessages.SETTINGS_MENU}"
        )
        return StateUpdate(State.SETTINGS_MENU, null)
    }

    private suspend fun executePendingExternalIntent(
        pendingIntent: PendingExternalIntent
    ): StateUpdate {
        val result = pendingIntent.intentResult
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] action=DISPATCH " +
                "intent=${result.intent} originalText='${result.originalText}' " +
                "source=STORED_NLP_RESULT nlpInferenceCountForOriginalText=1"
        )
        return when (result.intent) {
            Intent.TEXT_RECOGNITION -> handleTextRecognitionDirectly()
            Intent.OBJECT_DETECTION -> handleObjectDetectionRequest(result.originalText)
            Intent.MEASURE_DISTANCE ->
                handleUnresolvedCommand(classifierLabel = Intent.MEASURE_DISTANCE)
            else -> error("Intent ${result.intent} is not external to settings")
        }
	}

	suspend fun handleSettingsChoice(final: String): StateUpdate {
		return settingsHandler.handleSettingsChoice(final, lastDialogContext) { newJson ->
            lastDialogContext = newJson
        }
    }

    suspend fun handleSettingsAction(final: String): StateUpdate {
		return settingsHandler.handleSettingsAction(final, lastDialogContext) { newJson ->
            lastDialogContext = newJson
        }
    }

    suspend fun handleSettingsExternalConfirmation(final: String): StateUpdate {
        val pendingIntent = PendingExternalIntentCodec.decode(lastDialogContext)
        if (pendingIntent == null) {
            Log.e(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] " +
                    "outcome=INVALID_OR_MISSING action=RETURN_TO_SETTINGS_MENU"
            )
            lastDialogContext = null
            speechOutputHandler.speakAndHandleUi(
                "Der vorgemerkte Befehl ist nicht mehr verfügbar. ${LocalInteractionMessages.SETTINGS_MENU}"
            )
            return StateUpdate(State.SETTINGS_MENU, null)
        }

        // Do not run yes/no through the ten intent classes. This explicit
        // confirmation state is owned exclusively by the local confirmation model.
        return when (contextSwitchConfirmation.evaluate(final, pendingIntent)) {
            ContextSwitchConfirmationResult.APPROVED -> {
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] " +
                        "evaluator=LOCAL_CONFIRMATION_MODEL apiCalled=false " +
                        "decision=ACCEPT confirmation=APPROVED " +
                        "intent=${pendingIntent.intentResult.intent} " +
                        "action=EXECUTE_STORED_RESULT pendingContextRetained=false " +
                        "nlpReclassified=false"
                )
                lastDialogContext = null
                executePendingExternalIntent(pendingIntent)
            }

            ContextSwitchConfirmationResult.REJECTED -> {
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] " +
                        "evaluator=LOCAL_CONFIRMATION_MODEL apiCalled=false " +
                        "decision=REJECT confirmation=REJECTED " +
                        "action=STAY_IN_SETTINGS_MENU nextState=SETTINGS_MENU " +
                        "pendingExternalIntent=cleared"
                )
                lastDialogContext = null
                speechOutputHandler.speakAndHandleUi(
                    "Okay, Sie bleiben in den Einstellungen. ${LocalInteractionMessages.SETTINGS_MENU}"
                )
                StateUpdate(State.SETTINGS_MENU, null)
            }

			ContextSwitchConfirmationResult.UNKNOWN -> {
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] " +
                        "evaluator=LOCAL_CONFIRMATION_MODEL apiCalled=false " +
                        "decision=UNKNOWN confirmation=UNKNOWN action=REQUEST_CLARIFICATION " +
                        "nextState=SETTINGS_EXTERNAL_CONFIRMATION pendingContextRetained=true"
                )
				speechOutputHandler.speakAndHandleUi(
                    "Ich konnte die Bestätigung nicht eindeutig zuordnen. " +
                        "Bitte antworten Sie mit Ja oder Nein."
                )
                StateUpdate(State.SETTINGS_EXTERNAL_CONFIRMATION, lastDialogContext)
            }

            ContextSwitchConfirmationResult.FAILED -> {
                Log.w(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] " +
                        "evaluator=LOCAL_CONFIRMATION_MODEL apiCalled=false " +
                        "decision=FAILED confirmation=FAILED action=KEEP_PENDING_CONTEXT " +
                        "nextState=SETTINGS_EXTERNAL_CONFIRMATION pendingContextRetained=true"
                )
                StateUpdate(State.SETTINGS_EXTERNAL_CONFIRMATION, lastDialogContext)
            }
        }
    }
}
