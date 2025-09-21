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

    private val objectPositionClassifier = ObjectPositionClassifier()


    // Performance optimizations
    private val promptCache = mutableMapOf<String, String>()

    fun isStreaming(): Boolean = streamingHandler.isStreaming()

    fun getStreamingHandler(): LLMStreamingHandler = streamingHandler

    private fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

    suspend fun handleIdle(final: String): StateUpdate {
        val jsonResponse = generateLlmResponse(final, true) ?: return StateUpdate(State.IDLE, null)
        logDebugInfo(final, jsonResponse)

        return when (jsonParser.parseRequestedFunction(jsonResponse)) {
            RequestedFunction.OBJECT_DETECTION -> handleObjectDetectionRequest(jsonResponse)
            RequestedFunction.TEXT_RECOGNITION -> handleTextRecognitionRequest()
            RequestedFunction.SETTINGS -> handleSettingsRequest(jsonResponse)
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
            streamingHandler.speakAndHandleUi("Entschuldigung, die Texterkennung konnte nicht durchgefÃ¼hrt werden.")
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

    private suspend fun handleSettingsRequest(jsonResponse: String): StateUpdate {
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
        return settingsHandler.handleSettingsMenu(final, lastLlmJsonResponse) { newJson ->
            lastLlmJsonResponse = newJson
        }
    }

    suspend fun handleSettingsChoice(final: String): StateUpdate {
        return settingsHandler.handleSettingsChoice(final, lastLlmJsonResponse) { newJson ->
            lastLlmJsonResponse = newJson
        }
    }

    suspend fun handleSettingsAction(final: String): StateUpdate {
        return settingsHandler.handleSettingsAction(final, lastLlmJsonResponse) { newJson ->
            lastLlmJsonResponse = newJson
        }
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