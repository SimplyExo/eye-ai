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
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects.ObjectDetectionHandler
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.RequestedFunction
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.SettingsHandler
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import kotlinx.coroutines.delay
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects.ObjectPositionClassifier
import com.algorithmic_alliance.eyeaiapp.nlp.NLPModel
import com.algorithmic_alliance.eyeaiapp.nlp.OCRToText
import com.algorithmic_alliance.eyeaiapp.AIModelData

class StateMachine(
    private val eyeAIApp: EyeAIApp,
    private val textToSpeechInstance: TextToSpeechInstance,
    private var lastLlmJsonResponse: String?,
    private val setLlmResponseText: (String) -> Unit,
    private val appendLlmResponseText: (String) -> Unit,
    private val cameraFrameAnalyzer: CameraFrameAnalyzer? = null,
    private val onStreamingComplete: () -> Unit = {}
) {

    private val streamingHandler = LLMStreamingHandler(textToSpeechInstance, setLlmResponseText,appendLlmResponseText, eyeAIApp, onStreamingComplete)
    private val jsonParser = JsonParser()

    // SettingsHandler
    private val settingsHandler = SettingsHandler(
        textToSpeechInstance,
        jsonParser,
        eyeAIApp,
        ::generateLlmResponse,
        streamingHandler::speakAndHandleUi
    )

    private val objectPositionClassifier = ObjectPositionClassifier()
    private val ocrToText = OCRToText()

    // NLP model integration
    private val nlpModel: NLPModel? = eyeAIApp.nlpModel

    // Using LLM if less than 60%
    private val nlpConfidenceThreshold = 0.6f


    private val promptCache = mutableMapOf<String, String>()


    private val settingsIntents = setOf(
        NLPModel.NLPClasses.CHANGE_SPEECH_SPEED,
        NLPModel.NLPClasses.CHANGE_SPEAKER,
        NLPModel.NLPClasses.OPEN_SETTINGS,
        NLPModel.NLPClasses.SET_FREQUENCY,
        NLPModel.NLPClasses.SET_BPS
    )


    private val validSettingsIntents = setOf(
        NLPModel.NLPClasses.CHANGE_SPEECH_SPEED,
        NLPModel.NLPClasses.CHANGE_SPEAKER,
        NLPModel.NLPClasses.SET_FREQUENCY,
        NLPModel.NLPClasses.SET_BPS,
        NLPModel.NLPClasses.ABORT
    )

    private val exitKeywords = setOf(
        "verlassen", "stopp", "abbruch", "stop", "exit", "quit",
        "beenden", "raus", "zurück", "abbrechen", "cancel", "zumachen", "exit", "verlasse einstellungen", "schließ das", "home", "startseite", "ich will raus", "will hier raus"
    )

    private fun checkForExitKeywords(input: String): Boolean {
        val lowerInput = input.lowercase().trim()
        return exitKeywords.any { keyword ->
            lowerInput.contains(keyword) || lowerInput == keyword
        }
    }

    private suspend fun handleExitFromSettings(): StateUpdate {
        val syntheticLeave = createLeaveSettingsJson()
        streamingHandler.speakAndHandleUi("Möchten Sie die Einstellungen wirklich verlassen?")
        lastLlmJsonResponse = syntheticLeave
        return StateUpdate(State.SETTINGS_ACTION, syntheticLeave)
    }

    fun isStreaming(): Boolean = streamingHandler.isStreaming()

    fun getStreamingHandler(): LLMStreamingHandler = streamingHandler

    private fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

    suspend fun handleIdle(final: String): StateUpdate {
        // Intent with nlp if confidence is greater than 50%
        val (nlpIntent, confidence) = classifyIntentWithNLPAndConfidence(final)

        Log.d(EyeAIApp.APP_LOG_TAG, "NLP classified intent: $nlpIntent with confidence: ${String.format("%.2f", confidence * 100)}% for input: '$final'")

        // fallback to llm
        if (nlpIntent == null) {
            Log.d(EyeAIApp.APP_LOG_TAG, "NLP confidence below threshold (${String.format("%.2f", confidence * 100)}%), falling back to LLM")
            return handleWithLLMFallback(final)
        }

        return when (nlpIntent) {
            NLPModel.NLPClasses.TEXT_RECOGNITION -> handleTextRecognitionDirectly()
            NLPModel.NLPClasses.OBJECT_DETECTION -> handleObjectDetectionWithLLM(final)
            NLPModel.NLPClasses.MEASURE_DISTANCE -> handleMeasureDistanceWithLLM(final)
            in settingsIntents -> handleSettingsRequest()
            NLPModel.NLPClasses.REDIRECT_TO_LLM, NLPModel.NLPClasses.ABORT -> handleWithLLMFallback(final)
            else -> handleWithLLMFallback(final)
        }
    }

    private fun classifyIntentWithNLPAndConfidence(input: String): Pair<NLPModel.NLPClasses?, Float> {
        return try {
            val results = nlpModel?.runInferenceWithAllResults(input)
            if (results == null) return Pair(null, 0f)

            // Get intent with highest confidence
            val intentConfidences = listOf(
                NLPModel.NLPClasses.TEXT_RECOGNITION to results.TEXT_RECOGNITION,
                NLPModel.NLPClasses.OBJECT_DETECTION to results.OBJECT_DETECTION,
                NLPModel.NLPClasses.CHANGE_SPEECH_SPEED to results.CHANGE_SPEECH_SPEED,
                NLPModel.NLPClasses.CHANGE_SPEAKER to results.CHANGE_SPEAKER,
                NLPModel.NLPClasses.REDIRECT_TO_LLM to results.REDIRECT_TO_LLM,
                NLPModel.NLPClasses.OPEN_SETTINGS to results.OPEN_SETTINGS,
                NLPModel.NLPClasses.SET_FREQUENCY to results.SET_FREQUENCY,
                NLPModel.NLPClasses.SET_BPS to results.SET_BPS,
                NLPModel.NLPClasses.MEASURE_DISTANCE to results.MEASURE_DISTANCE,
                NLPModel.NLPClasses.ABORT to results.ABORT
            )

            val bestMatch = intentConfidences.maxByOrNull { it.second }
            if (bestMatch != null && bestMatch.second >= nlpConfidenceThreshold) {
                Pair(bestMatch.first, bestMatch.second)
            } else {
                Pair(null, bestMatch?.second ?: 0f)
            }
        } catch (e: Exception) {
            Log.e(EyeAIApp.APP_LOG_TAG, "NLP classification with confidence failed", e)
            Pair(null, 0f)
        }
    }

    // with confidence levels now
    private fun classifyIntentWithNLPFallbackForSettings(input: String): Pair<NLPModel.NLPClasses?, Float> {
        return try {
            val results = nlpModel?.runInferenceWithAllResults(input)
            if (results == null) return Pair(null, 0f)

            // Convert results to list of intent and confidence
            val intentConfidences = listOf(
                NLPModel.NLPClasses.TEXT_RECOGNITION to results.TEXT_RECOGNITION,
                NLPModel.NLPClasses.OBJECT_DETECTION to results.OBJECT_DETECTION,
                NLPModel.NLPClasses.CHANGE_SPEECH_SPEED to results.CHANGE_SPEECH_SPEED,
                NLPModel.NLPClasses.CHANGE_SPEAKER to results.CHANGE_SPEAKER,
                NLPModel.NLPClasses.REDIRECT_TO_LLM to results.REDIRECT_TO_LLM,
                NLPModel.NLPClasses.OPEN_SETTINGS to results.OPEN_SETTINGS,
                NLPModel.NLPClasses.SET_FREQUENCY to results.SET_FREQUENCY,
                NLPModel.NLPClasses.SET_BPS to results.SET_BPS,
                NLPModel.NLPClasses.MEASURE_DISTANCE to results.MEASURE_DISTANCE,
                NLPModel.NLPClasses.ABORT to results.ABORT
            ).sortedByDescending { it.second }

            // Ignore non-settings intents, use next best if first is invalid
            for ((intent, confidence) in intentConfidences) {
                if (intent in validSettingsIntents && confidence >= nlpConfidenceThreshold) {
                    return Pair(intent, confidence)
                }
            }

            // get best with confidence
            val bestValidIntent = intentConfidences.find { it.first in validSettingsIntents }
            return Pair(null, bestValidIntent?.second ?: 0f)
        } catch (e: Exception) {
            Log.e(EyeAIApp.APP_LOG_TAG, "NLP classification with fallback failed", e)
            Pair(null, 0f)
        }
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

    private suspend fun handleSettingsRequest(): StateUpdate {
        streamingHandler.speakAndHandleUi(LLM.Companion.SNIPPET_SETTINGS)
        return StateUpdate(State.SETTINGS_MENU, null)
    }

    private suspend fun handleWithLLMFallback(final: String): StateUpdate {
        // Use LLM as fallback for REDIRECT_TO_LLM, ABORT, or when NLP fails
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

        //keyword search
        if (checkForExitKeywords(final)) {
            Log.d(EyeAIApp.APP_LOG_TAG, "Exit keyword detected in settings menu: '$final'")
            return handleExitFromSettings()
        }

        // Use NLP for intent classification in settings with confidence check
        val (nlpIntent, confidence) = classifyIntentWithNLPFallbackForSettings(final)

        Log.d(EyeAIApp.APP_LOG_TAG, "Settings menu NLP classified intent: $nlpIntent with confidence: ${String.format("%.2f", confidence * 100)}% for input: '$final'")

        // If confidence is below, back to LLM
        if (nlpIntent == null) {
            Log.d(EyeAIApp.APP_LOG_TAG, "Settings NLP confidence below threshold (${String.format("%.2f", confidence * 100)}%), falling back to LLM")
            return settingsHandler.handleSettingsMenu(final, lastLlmJsonResponse) { newJson ->
                lastLlmJsonResponse = newJson
            }
        }

        return when (nlpIntent) {
            NLPModel.NLPClasses.CHANGE_SPEECH_SPEED -> {
                streamingHandler.speakAndHandleUi(LLM.Companion.SNIPPET_TTS_SPEED)
                lastLlmJsonResponse = createSyntheticSettingsJson("tts_speed")
                StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
            }

            NLPModel.NLPClasses.CHANGE_SPEAKER -> {
                streamingHandler.speakAndHandleUi(LLM.Companion.SNIPPET_VOICE)
                lastLlmJsonResponse = createSyntheticSettingsJson("voice")
                StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
            }

            NLPModel.NLPClasses.SET_FREQUENCY -> {
                streamingHandler.speakAndHandleUi(LLM.Companion.SNIPPET_FREQUENCY)
                lastLlmJsonResponse = createSyntheticSettingsJson("frequency")
                StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
            }

            NLPModel.NLPClasses.SET_BPS -> {
                streamingHandler.speakAndHandleUi(LLM.Companion.SNIPPET_BPS)
                lastLlmJsonResponse = createSyntheticSettingsJson("bps")
                StateUpdate(State.SETTINGS_CHOICE, lastLlmJsonResponse)
            }

            NLPModel.NLPClasses.ABORT -> {
                val syntheticLeave = createLeaveSettingsJson()
                streamingHandler.speakAndHandleUi("Möchten Sie die Einstellungen wirklich verlassen?")
                lastLlmJsonResponse = syntheticLeave
                StateUpdate(State.SETTINGS_ACTION, syntheticLeave)
            }

            else -> {
                settingsHandler.handleSettingsMenu(final, lastLlmJsonResponse) { newJson ->
                    lastLlmJsonResponse = newJson
                }
            }
        }
    }

    suspend fun handleSettingsChoice(final: String): StateUpdate {

        if (checkForExitKeywords(final)) {
            Log.d(EyeAIApp.APP_LOG_TAG, "Exit keyword detected in settings choice: '$final'")
            return handleExitFromSettings()
        }

        return settingsHandler.handleSettingsChoice(final, lastLlmJsonResponse) { newJson ->
            lastLlmJsonResponse = newJson
        }
    }

    suspend fun handleSettingsAction(final: String): StateUpdate {

        if (checkForExitKeywords(final)) {
            Log.d(EyeAIApp.APP_LOG_TAG, "Exit keyword detected in settings choice: '$final'")
            return handleExitFromSettings()
        }

        return settingsHandler.handleSettingsAction(final, lastLlmJsonResponse) { newJson ->
            lastLlmJsonResponse = newJson
        }
    }

    private fun createSyntheticSettingsJson(settingType: String): String {
        return """{"requested_function": "settings", "setting_intent": "$settingType"}"""
    }

    private fun createLeaveSettingsJson(): String {
        return """{"changed_settings": [{"leave": true}]}"""
    }

    private suspend fun generateLlmResponse(prompt: String, structured: Boolean): String? {
        val promptTrimmed = prompt.trim()
        if (promptTrimmed.isEmpty()) return null

        // Cache check
        val cacheKey = "$promptTrimmed:$structured"
        promptCache[cacheKey]?.let { return it }

        val start = System.nanoTime()
        return try {
            val result = eyeAIApp.llm!!.generate(promptTrimmed, structured)
            Log.d(EyeAIApp.APP_LOG_TAG, "LLM generate (non-stream) END duration=${elapsedMs(start)} ms")

            // Cache result (limit cache size)
            if (promptCache.size < 50) {
                promptCache[cacheKey] = result
            }

            result
        } catch (e: Exception) {
            Log.e(EyeAIApp.APP_LOG_TAG, "LLM generate (non-stream) EXCEPTION after ${elapsedMs(start)} ms", e)
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
