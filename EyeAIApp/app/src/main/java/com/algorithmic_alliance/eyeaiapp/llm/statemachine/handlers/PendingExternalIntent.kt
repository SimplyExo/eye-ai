package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import com.algorithmic_alliance.eyeaiapp.nlp.IntentResult
import org.json.JSONArray
import org.json.JSONObject

data class PendingExternalIntent(val intentResult: IntentResult) {
	init {
		require(SettingsMenuIntentRouter.isExternalIntent(intentResult.intent)) {
			"Only global intents can be pending outside SETTINGS_MENU"
		}
	}
}

/** Persists all ten original NLP scores across MainActivity's StateMachine instances. */
object PendingExternalIntentCodec {
	private const val CONTEXT_TYPE_KEY = "context_type"
	private const val CONTEXT_TYPE_VALUE = "pending_external_intent"
	private const val PENDING_KEY = "pending_external_intent"
	private const val INTENT_KEY = "intent"
	private const val CONFIDENCE_KEY = "confidence"
	private const val ORIGINAL_TEXT_KEY = "original_text"
	private const val PROBABILITIES_KEY = "probabilities"

	fun encode(pendingIntent: PendingExternalIntent): String {
		val result = pendingIntent.intentResult
		return JSONObject().apply {
			put(CONTEXT_TYPE_KEY, CONTEXT_TYPE_VALUE)
			put(PENDING_KEY, JSONObject().apply {
				put(INTENT_KEY, result.intent.name)
				put(CONFIDENCE_KEY, result.confidence.toDouble())
				put(ORIGINAL_TEXT_KEY, result.originalText)
				put(PROBABILITIES_KEY, JSONArray().apply {
					result.probabilities.forEach { put(it.toDouble()) }
				})
			})
		}.toString()
	}

	fun decode(contextJson: String?): PendingExternalIntent? {
		if (contextJson == null) return null
		return try {
			val root = JSONObject(contextJson)
			if (root.optString(CONTEXT_TYPE_KEY) != CONTEXT_TYPE_VALUE) return null
			val pendingJson = root.getJSONObject(PENDING_KEY)
			val probabilitiesJson = pendingJson.getJSONArray(PROBABILITIES_KEY)
			val probabilities = FloatArray(probabilitiesJson.length()) { index ->
				probabilitiesJson.getDouble(index).toFloat()
			}
			val intentResult = IntentResult(
				intent = Intent.valueOf(pendingJson.getString(INTENT_KEY)),
				confidence = pendingJson.getDouble(CONFIDENCE_KEY).toFloat(),
				originalText = pendingJson.getString(ORIGINAL_TEXT_KEY),
				probabilities = probabilities
			)
			PendingExternalIntent(intentResult)
		} catch (_: Exception) {
			null
		}
	}
}

object PendingExternalIntentPresentation {
	fun confirmationQuestion(intent: Intent): String {
		val action = when (intent) {
			Intent.TEXT_RECOGNITION -> "die Texterkennung ausführen"
			Intent.OBJECT_DETECTION -> "die Objekterkennung ausführen"
			Intent.MEASURE_DISTANCE -> "die Entfernung messen"
			Intent.REDIRECT_TO_LLM -> "Ihre ursprüngliche Anfrage ausführen"
			else -> throw IllegalArgumentException("Intent $intent is not external to settings")
		}
		return "Sie befinden sich noch in den Einstellungen. " +
			"Möchten Sie die Einstellungen verlassen und $action?"
	}
}
