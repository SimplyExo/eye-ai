package com.algorithmic_alliance.eyeaiapp.llm.statemachine

import android.util.Log
import android.widget.TextView
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.MainActivity.State
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio.GoogleAIStudioLLM
import com.algorithmic_alliance.eyeaiapp.llm.LLM
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.JsonParser
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.LLMStreamingHandler
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.ContextSwitchConfirmationResult
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.GeminiContextSwitchConfirmation
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.PendingExternalIntent
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.PendingExternalIntentCodec
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.PendingExternalIntentPresentation
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects.ObjectDetectionHandler
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.RequestedFunction
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.SettingIntent
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
    private var lastLlmJsonResponse: String?,
    private val llmResponseText: TextView?,
    private val cameraFrameAnalyzer: CameraFrameAnalyzer? = null,
    private val onStreamingComplete: () -> Unit = {}
) {

    private val streamingHandler = LLMStreamingHandler(textToSpeechInstance, llmResponseText, eyeAIApp, onStreamingComplete)
    private val jsonParser = JsonParser()

    // SettingsHandler
    private val settingsHandler = SettingsHandler(
        textToSpeechInstance,
        jsonParser,
        eyeAIApp,
        ::generateLlmResponse,
        streamingHandler::speakAndHandleUi
    )
    private val contextSwitchConfirmation = GeminiContextSwitchConfirmation(
        jsonParser = jsonParser,
        trace = { message -> Log.d(EyeAIApp.APP_LOG_TAG, message) },
        generateLlmResponse = ::generateLlmResponse
    )

    private val objectPositionClassifier = ObjectPositionClassifier()
    private val ocrToText = OCRToText()

    // NLP model integration
    private val nlpModel: NLPModel? = eyeAIApp.nlpModel

    /** Complete, unfiltered NLP V2 result for future state-aware consumers. */
    var lastIntentResult: IntentResult? = null
        private set

    // Using LLM if less than 60%
    private val nlpConfidenceThreshold = 0.6f


    private val promptCache = mutableMapOf<String, String>()

    fun isStreaming(): Boolean = streamingHandler.isStreaming()

    fun getStreamingHandler(): LLMStreamingHandler = streamingHandler

    private fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

    suspend fun handleIdle(final: String): StateUpdate {
        val intentResult = classifyIntentWithNLP(final)
        val confidence = intentResult?.confidence ?: 0f
        val nlpIntent = intentResult?.intent?.takeIf { confidence >= nlpConfidenceThreshold }

        // fallback to llm
        if (nlpIntent == null) {
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][ROUTE] classifier=NLP_V2 accepted=false " +
                    "confidence=${formatPercentage(confidence)} threshold=${formatPercentage(nlpConfidenceThreshold)} " +
                    "nextEvaluator=GEMINI_API role=INTENT_FALLBACK"
            )
            return handleWithLLMFallback(final)
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
                    "GEMINI_API",
                    "DIRECT_SETTINGS_PARAMETER_EXTRACTION"
                )
                handleDirectSettingsRequest(settingsRoute)
            }

            SettingsIntentRoute.NotSettings -> when (nlpIntent) {
                Intent.TEXT_RECOGNITION -> {
                    logNlpRoute(nlpIntent, "LOCAL_OCR_PIPELINE", "TEXT_RECOGNITION")
                    handleTextRecognitionDirectly()
                }

                Intent.OBJECT_DETECTION -> {
                    logNlpRoute(nlpIntent, "GEMINI_API", "OBJECT_QUERY_EVALUATION")
                    handleObjectDetectionWithLLM(final)
                }

                Intent.MEASURE_DISTANCE -> {
                    logNlpRoute(nlpIntent, "GEMINI_API", "DISTANCE_REQUEST_EVALUATION")
                    handleMeasureDistanceWithLLM(final)
                }

                Intent.REDIRECT_TO_LLM, Intent.ABORT -> {
                    logNlpRoute(nlpIntent, "GEMINI_API", "INTENT_FALLBACK")
                    handleWithLLMFallback(final)
                }

                else -> {
                    logNlpRoute(nlpIntent, "GEMINI_API", "INTENT_FALLBACK")
                    handleWithLLMFallback(final)
                }
            }
        }
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

    private fun isConfidentAbort(intentResult: IntentResult?): Boolean =
        intentResult?.intent == Intent.ABORT &&
            intentResult.confidence >= nlpConfidenceThreshold

    private suspend fun abortSettingsFlow(sourceState: State): StateUpdate {
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][SETTINGS_ABORT] classifier=NLP_V2 " +
                "state=$sourceState outcome=RETURN_TO_IDLE pendingExternalIntent=cleared"
        )
        lastLlmJsonResponse = null
        streamingHandler.speakAndHandleUi("Okay, ich habe den Einstellungsdialog abgebrochen.")
        return StateUpdate(State.IDLE, null)
    }

    private fun classifyAbortForSettingsChoice(input: String): Boolean {
        val result = classifyIntentWithNLP(input)
        val accepted = isConfidentAbort(result)
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][SETTINGS_ABORT_GATE] classifier=NLP_V2 " +
                "state=SETTINGS_CHOICE top1=${result?.intent} " +
                "confidence=${formatPercentage(result?.confidence ?: 0f)} " +
                "threshold=${formatPercentage(nlpConfidenceThreshold)} accepted=$accepted"
        )
        return accepted
    }

    private suspend fun handleTextRecognitionDirectly(): StateUpdate {
        val ocrSuccess = cameraFrameAnalyzer?.runOcrAnalysis() ?: false

        if (!ocrSuccess) {
            Log.d(EyeAIApp.APP_LOG_TAG, "OCR analysis failed")
            streamingHandler.speakAndHandleUi("Entschuldigung, die Texterkennung konnte nicht durchgeführt werden.")
            return StateUpdate(State.IDLE, null)
        }

        delay(200) // Wait for OCR result to be available

        // Get OCR boxes and convert Array to List
        val ocrBoxes = AIModelData.ocrBoxes.get()

        if (ocrBoxes.isNullOrEmpty()) {
            Log.d(EyeAIApp.APP_LOG_TAG, "No OCR boxes available after OCR analysis")
            streamingHandler.speakAndHandleUi("Entschuldigung, es wurde kein Text erkannt.")
            return StateUpdate(State.IDLE, null)
        }

        // Convert Array to List for OCRToText
        val ocrBoxesList = ocrBoxes.toList()

        // Use OCRToText to generate description
        val readableText = ocrToText.generateReadableText(ocrBoxesList)

        if (readableText.isEmpty()) {
            Log.d(EyeAIApp.APP_LOG_TAG, "OCRToText generated empty text")
            streamingHandler.speakAndHandleUi("Entschuldigung, ich konnte keinen sinnvollen Text erkennen.")
            return StateUpdate(State.IDLE, null)
        }

        Log.d(EyeAIApp.APP_LOG_TAG, "OCRToText generated: $readableText")
        streamingHandler.speakAndHandleUi(readableText)

        return StateUpdate(State.IDLE, null)
    }

    private suspend fun handleObjectDetectionWithLLM(final: String): StateUpdate {

        val jsonResponse = generateLlmResponse(final, true) ?: return StateUpdate(State.IDLE, null)
        logDebugInfo(final, jsonResponse)
        return handleObjectDetectionRequest(jsonResponse)
    }

    private suspend fun handleMeasureDistanceWithLLM(final: String): StateUpdate {

        val jsonResponse = generateLlmResponse(final, true) ?: return StateUpdate(State.IDLE, null)
        logDebugInfo(final, jsonResponse)
        return handleNoneRequest(jsonResponse, final)
    }

    private suspend fun openGuidedSettingsMenu(): StateUpdate {
        streamingHandler.speakAndHandleUi(LLM.Companion.SNIPPET_SETTINGS)
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
        lastLlmJsonResponse = settingsContext

        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][HANDOFF] classifier=NLP_V2 " +
                "intent=${intentResult.intent} evaluator=GEMINI_API " +
                "role=SETTINGS_PARAMETER_EXTRACTION settingIntent=${route.settingIntent} " +
                "originalText='${intentResult.originalText}'"
        )

        return settingsHandler.handleSettingsChoice(
            input = intentResult.originalText,
            currentJson = settingsContext
        ) { newJson ->
            lastLlmJsonResponse = newJson
        }
    }

    private suspend fun handleWithLLMFallback(final: String): StateUpdate {
        // Use LLM as fallback for REDIRECT_TO_LLM, ABORT, or when NLP fails
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][Gemini API][EVALUATE] role=INTENT_FALLBACK input='$final'"
        )
        val jsonResponse = generateLlmResponse(final, true) ?: return StateUpdate(State.IDLE, null)
        logDebugInfo(final, jsonResponse)

        return when (jsonParser.parseRequestedFunction(jsonResponse)) {
            RequestedFunction.OBJECT_DETECTION -> handleObjectDetectionRequest(jsonResponse)
            RequestedFunction.TEXT_RECOGNITION -> handleTextRecognitionRequest()
            RequestedFunction.SETTINGS -> handleSettingsRequestFromLLM(jsonResponse)
            RequestedFunction.NONE -> handleNoneRequest(jsonResponse, final)
        }
    }

    private suspend fun handleObjectDetectionRequest(jsonResponse: String): StateUpdate {
        val germanObjectQuery = jsonParser.parseObjectQuery(jsonResponse)?.trim() ?: ""
        Log.d(EyeAIApp.APP_LOG_TAG, "German object query from LLM: '$germanObjectQuery'")
        Log.d(EyeAIApp.APP_LOG_TAG, "JSON response: $jsonResponse")

        if (germanObjectQuery.isBlank()) {
            Log.w(EyeAIApp.APP_LOG_TAG, "Object query is blank")
            val availableGermanObjects = ObjectDetectionHandler.getGermanObjectLabelsForLLM()
            Log.d(EyeAIApp.APP_LOG_TAG, "Available German objects: $availableGermanObjects")

            if (availableGermanObjects.isNotEmpty()) {
                val objectList = availableGermanObjects.take(5).joinToString(", ")
                streamingHandler.speakAndHandleUi(
                    "Bitte nennen Sie ein spezifisches Objekt, nach dem Sie suchen möchten. " +
                        "Verfügbare Objekte: $objectList"
                )
            } else {
                streamingHandler.speakAndHandleUi("Entschuldigung, ich konnte gerade keine Objekte erkennen.")
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
                streamingHandler.speakAndHandleUi(description)
            }

            is ObjectDetectionHandler.ObjectDetectionResult.ObjectNotFound -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "Object NOT FOUND. Available: ${result.availableObjects}")
                streamingHandler.speakAndHandleUi(
                    "Entschuldigung, das Objekt '$germanObjectQuery' konnte ich nicht finden. " +
                        "Ich sehe aber folgende Objekte: ${result.availableObjects.joinToString(", ")}."
                )
            }

            is ObjectDetectionHandler.ObjectDetectionResult.NoObjectsFound -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "No objects found")
                streamingHandler.speakAndHandleUi("Entschuldigung, ich konnte gerade keine Objekte erkennen.")
            }

            is ObjectDetectionHandler.ObjectDetectionResult.DepthDataUnavailable -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "Depth data unavailable")
                streamingHandler.speakAndHandleUi("Entschuldigung, die Tiefenerkennung ist derzeit nicht verfügbar.")
            }

            is ObjectDetectionHandler.ObjectDetectionResult.DepthDataInvalid -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "Depth data invalid")
                streamingHandler.speakAndHandleUi("Entschuldigung, die Tiefendaten haben eine unerwartete Größe.")
            }

            is ObjectDetectionHandler.ObjectDetectionResult.NoKnownObjectsFound -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "No known objects found")
                streamingHandler.speakAndHandleUi("Ich konnte leider keine bekannten Objekte erkennen.")
            }

            is ObjectDetectionHandler.ObjectDetectionResult.NoQueryProvided -> {
                Log.d(EyeAIApp.APP_LOG_TAG, "No query provided")
                streamingHandler.speakAndHandleUi("Bitte nennen Sie ein spezifisches Objekt.")
            }
        }

        return StateUpdate(State.IDLE, null)
    }

    private suspend fun handleTextRecognitionRequest(): StateUpdate {
        val ocrSuccess = cameraFrameAnalyzer?.runOcrAnalysis() ?: false

        if (!ocrSuccess) {
            Log.d(EyeAIApp.APP_LOG_TAG, "OCR analysis failed")
            streamingHandler.speakAndHandleUi("Entschuldigung, die Texterkennung konnte nicht durchgeführt werden.")
            return StateUpdate(State.IDLE, null)
        }

        delay(200) // Wait for OCR result to be available

        val ocrText = eyeAIApp.ocrModel.lastResult.trim()

        if (ocrText.isEmpty()) {
            Log.d(EyeAIApp.APP_LOG_TAG, "No OCR text available after on-demand analysis")
            streamingHandler.speakAndHandleUi("Entschuldigung, es wurde kein Text erkannt.")
            return StateUpdate(State.IDLE, null)
        }

        val prompt = eyeAIApp.llm!!.buildOcrPrompt(ocrText)

        if (prompt.trim().isEmpty()) {
            Log.w(EyeAIApp.APP_LOG_TAG, "OCR prompt is empty")
            streamingHandler.speakAndHandleUi("Entschuldigung, ich konnte keinen sinnvollen Text erkennen.")
            return StateUpdate(State.IDLE, null)
        }

        streamingHandler.generateAndStreamResponse(eyeAIApp.llm as GoogleAIStudioLLM, prompt)
        return StateUpdate(State.IDLE, null)
    }

    private suspend fun handleSettingsRequestFromLLM(jsonResponse: String): StateUpdate {
        streamingHandler.speakAndHandleUi(LLM.Companion.SNIPPET_SETTINGS)
        lastLlmJsonResponse = jsonResponse
        return StateUpdate(State.SETTINGS_MENU, lastLlmJsonResponse)
    }

    private suspend fun handleNoneRequest(jsonResponse: String, final: String): StateUpdate {
        val direct = jsonParser.parseInteractionText(jsonResponse)
        return if (!direct.isNullOrBlank()) {
            streamingHandler.speakAndHandleUi(direct)
            StateUpdate(State.IDLE, null)
        } else {
            val fallbackResponse = generateLlmResponse(final, false) ?: jsonResponse
            streamingHandler.speakAndHandleUi(fallbackResponse)
            StateUpdate(State.IDLE, null)
        }
    }

    suspend fun handleSettingsMenu(final: String): StateUpdate {
        val intentResult = classifyIntentWithNLP(final)
        if (intentResult == null) {
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][SETTINGS_ROUTE] classifier=NLP_V2 " +
                    "outcome=UNAVAILABLE nextEvaluator=GEMINI_API role=GUIDED_SETTINGS_INTENT"
            )
            return handleSettingsMenuWithGeminiFallback(final)
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
                abortSettingsFlow(State.SETTINGS_MENU)

            SettingsMenuIntentRoute.AlreadyInSettings ->
                remindUserSettingsAreAlreadyOpen()

            is SettingsMenuIntentRoute.GeminiFallback ->
                handleSettingsMenuWithGeminiFallback(final)
        }
    }

    private suspend fun handleLocalSettingsMenuIntent(intent: Intent): StateUpdate {
        val (settingIntent, prompt) = when (intent) {
            Intent.CHANGE_SPEECH_SPEED -> SettingIntent.TTS_SPEED to LLM.Companion.SNIPPET_TTS_SPEED
            Intent.CHANGE_SPEAKER -> SettingIntent.VOICE to LLM.Companion.SNIPPET_VOICE
            Intent.SET_FREQUENCY -> SettingIntent.FREQUENCY to LLM.Companion.SNIPPET_FREQUENCY
            Intent.SET_BPS -> SettingIntent.BPS to LLM.Companion.SNIPPET_BPS
            else -> error("Intent $intent is not a concrete settings intent")
        }
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][SETTINGS_ROUTE] classifier=NLP_V2 " +
                "intent=$intent action=LOCAL_SETTINGS_SELECTION nextState=SETTINGS_CHOICE"
        )
        streamingHandler.speakAndHandleUi(prompt)
        lastLlmJsonResponse = jsonParser.createSettingsContext(settingIntent)
        return StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
    }

    private suspend fun requestExternalIntentContextSwitch(
        intentResult: IntentResult
    ): StateUpdate {
        val pendingIntent = PendingExternalIntent(intentResult)
        val pendingContext = PendingExternalIntentCodec.encode(pendingIntent)
        lastLlmJsonResponse = pendingContext
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] action=STORE " +
                "intent=${intentResult.intent} confidence=${formatPercentage(intentResult.confidence)} " +
                "originalText='${intentResult.originalText}' probabilities=${intentResult.probabilities.size}"
        )
        streamingHandler.speakAndHandleUi(
            PendingExternalIntentPresentation.confirmationQuestion(intentResult.intent)
        )
        return StateUpdate(State.SETTINGS_EXTERNAL_CONFIRMATION, pendingContext)
    }

    private suspend fun remindUserSettingsAreAlreadyOpen(): StateUpdate {
        lastLlmJsonResponse = null
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][SETTINGS_ROUTE] classifier=NLP_V2 " +
                "intent=OPEN_SETTINGS action=REPLAY_EXISTING_MENU nestedFlowCreated=false"
        )
        streamingHandler.speakAndHandleUi(
            "Sie befinden sich bereits in den Einstellungen. ${LLM.Companion.SNIPPET_SETTINGS}"
        )
        return StateUpdate(State.SETTINGS_MENU, null)
    }

    private suspend fun handleSettingsMenuWithGeminiFallback(final: String): StateUpdate {
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][SETTINGS_ROUTE] classifier=NLP_V2 accepted=false " +
                "nextEvaluator=GEMINI_API role=GUIDED_SETTINGS_INTENT"
        )
        return settingsHandler.handleSettingsMenu(final, lastLlmJsonResponse) { newJson ->
            lastLlmJsonResponse = newJson
        }
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
            Intent.OBJECT_DETECTION -> handleObjectDetectionWithLLM(result.originalText)
            Intent.MEASURE_DISTANCE -> handleMeasureDistanceWithLLM(result.originalText)
            Intent.REDIRECT_TO_LLM -> handleWithLLMFallback(result.originalText)
            else -> error("Intent ${result.intent} is not external to settings")
        }
    }

    suspend fun handleSettingsChoice(final: String): StateUpdate {
        if (classifyAbortForSettingsChoice(final)) {
            return abortSettingsFlow(State.SETTINGS_CHOICE)
        }

        return settingsHandler.handleSettingsChoice(final, lastLlmJsonResponse) { newJson ->
            lastLlmJsonResponse = newJson
        }
    }

    suspend fun handleSettingsAction(final: String): StateUpdate {
        // The ten-class CNN has no yes/no classes. Gemini therefore keeps
        // confirmation ownership and distinguishes approval, rejection, and abort.
        return settingsHandler.handleSettingsAction(final, lastLlmJsonResponse) { newJson ->
            lastLlmJsonResponse = newJson
        }
    }

    suspend fun handleSettingsExternalConfirmation(final: String): StateUpdate {
        val pendingIntent = PendingExternalIntentCodec.decode(lastLlmJsonResponse)
        if (pendingIntent == null) {
            Log.e(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] " +
                    "outcome=INVALID_OR_MISSING action=RETURN_TO_SETTINGS_MENU"
            )
            lastLlmJsonResponse = null
            streamingHandler.speakAndHandleUi(
                "Der vorgemerkte Befehl ist nicht mehr verfügbar. ${LLM.Companion.SNIPPET_SETTINGS}"
            )
            return StateUpdate(State.SETTINGS_MENU, null)
        }

        // Do not run yes/no through the ten intent classes: on the frozen model
        // both answers can appear as ABORT. Gemini owns this confirmation state.
        return when (contextSwitchConfirmation.evaluate(final, pendingIntent)) {
            ContextSwitchConfirmationResult.APPROVED -> {
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] " +
                        "confirmation=APPROVED intent=${pendingIntent.intentResult.intent} " +
                        "action=EXECUTE_STORED_RESULT nlpReclassified=false"
                )
                lastLlmJsonResponse = null
                executePendingExternalIntent(pendingIntent)
            }

            ContextSwitchConfirmationResult.REJECTED -> {
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] " +
                        "confirmation=REJECTED action=STAY_IN_SETTINGS_MENU pendingExternalIntent=cleared"
                )
                lastLlmJsonResponse = null
                streamingHandler.speakAndHandleUi(
                    "Okay, Sie bleiben in den Einstellungen. ${LLM.Companion.SNIPPET_SETTINGS}"
                )
                StateUpdate(State.SETTINGS_MENU, null)
            }

            ContextSwitchConfirmationResult.ABORTED -> {
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] " +
                        "confirmation=ABORTED action=RETURN_TO_IDLE pendingExternalIntent=cleared"
                )
                abortSettingsFlow(State.SETTINGS_EXTERNAL_CONFIRMATION)
            }

            ContextSwitchConfirmationResult.FAILED -> {
                Log.w(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][PENDING_EXTERNAL_INTENT] " +
                        "confirmation=FAILED action=KEEP_PENDING_CONTEXT"
                )
                StateUpdate(State.SETTINGS_EXTERNAL_CONFIRMATION, lastLlmJsonResponse)
            }
        }
    }

    private suspend fun generateLlmResponse(prompt: String, structured: Boolean): String? {
        val promptTrimmed = prompt.trim()
        if (promptTrimmed.isEmpty()) return null

        // Cache check
        val cacheKey = "$promptTrimmed:$structured"
        promptCache[cacheKey]?.let { cachedResult ->
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][Gemini API][CACHE_HIT] apiCalled=false structured=$structured"
            )
            return cachedResult
        }

        val start = System.nanoTime()
        return try {
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][Gemini API][REQUEST] apiCalled=true " +
                    "model=${GoogleAIStudioLLM.MODEL_NAME} structured=$structured " +
                    "promptPreview='${promptTrimmed.take(160)}'"
            )
            val result = eyeAIApp.llm!!.generate(promptTrimmed, structured)
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][Gemini API][RESPONSE] model=${GoogleAIStudioLLM.MODEL_NAME} " +
                    "duration=${elapsedMs(start)}ms responsePreview='${result.take(240)}'"
            )

            // Cache result (limit cache size)
            if (promptCache.size < 50) {
                promptCache[cacheKey] = result
            }

            result
        } catch (e: Exception) {
            Log.e(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][Gemini API][RESPONSE] outcome=FAILED " +
                    "duration=${elapsedMs(start)}ms",
                e
            )
            streamingHandler.speakAndHandleUi("Entschuldigung, bei der Anfrage ist ein Fehler aufgetreten.")
            null
        }
    }

    private fun logDebugInfo(final: String, jsonResponse: String) {
        Log.d(EyeAIApp.APP_LOG_TAG, "handleIdle called with: '$final', parsed function: ${jsonParser.parseRequestedFunction(jsonResponse)}")
        Log.d(EyeAIApp.APP_LOG_TAG, "User input: '$final'")
        Log.d(EyeAIApp.APP_LOG_TAG, "LLM JSON response: $jsonResponse")
        Log.d(EyeAIApp.APP_LOG_TAG, "Parsed function: ${jsonParser.parseRequestedFunction(jsonResponse)}")
        Log.d(EyeAIApp.APP_LOG_TAG, "Parsed object query: '${jsonParser.parseObjectQuery(jsonResponse)}'")
    }
}
