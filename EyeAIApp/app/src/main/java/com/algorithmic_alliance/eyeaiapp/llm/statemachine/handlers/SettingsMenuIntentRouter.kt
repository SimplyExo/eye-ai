package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import com.algorithmic_alliance.eyeaiapp.nlp.IntentResult

/** Unmodified NLP evidence used for a SETTINGS_MENU routing decision. */
data class SettingsMenuIntentEvidence(
	val topIntent: Intent,
	val topConfidence: Float,
	val bestSettingsIntent: Intent?,
	val bestSettingsConfidence: Float
)

sealed class SettingsMenuIntentRoute {
	data class LocalSetting(
		val intent: Intent,
		val confidence: Float
	) : SettingsMenuIntentRoute()

	data class ExternalIntent(
		val intent: Intent,
		val confidence: Float
	) : SettingsMenuIntentRoute()

	data object Abort : SettingsMenuIntentRoute()
	data object AlreadyInSettings : SettingsMenuIntentRoute()
	data object Unresolved : SettingsMenuIntentRoute()
}

/**
 * Adds state context without masking or renormalizing the ten NLP V2 classes.
 * A confident global top-1 always wins over a weaker settings candidate.
 */
object SettingsMenuIntentRouter {
	private val concreteSettingsIntents = setOf(
		Intent.CHANGE_SPEECH_SPEED,
		Intent.CHANGE_SPEAKER,
		Intent.SET_FREQUENCY,
		Intent.SET_BPS,
		Intent.ABORT
	)

	private val externalIntents = setOf(
		Intent.TEXT_RECOGNITION,
		Intent.OBJECT_DETECTION,
		Intent.MEASURE_DISTANCE
	)

	fun route(
		intentResult: IntentResult,
		confidenceThreshold: Float
	): SettingsMenuIntentRoute = route(
		evidence = evidenceFrom(intentResult),
		confidenceThreshold = confidenceThreshold
	)

	/** Public evidence overload keeps conflict behavior directly unit-testable. */
	fun route(
		evidence: SettingsMenuIntentEvidence,
		confidenceThreshold: Float
	): SettingsMenuIntentRoute {
		if (evidence.topIntent == Intent.REDIRECT_TO_LLM) {
			return SettingsMenuIntentRoute.Unresolved
		}

		if (evidence.topConfidence >= confidenceThreshold) {
			when (evidence.topIntent) {
				in externalIntents -> return SettingsMenuIntentRoute.ExternalIntent(
					intent = evidence.topIntent,
					confidence = evidence.topConfidence
				)

				Intent.OPEN_SETTINGS -> return SettingsMenuIntentRoute.AlreadyInSettings
				Intent.ABORT -> return SettingsMenuIntentRoute.Abort
				else -> Unit
			}
		}

		val settingsIntent = evidence.bestSettingsIntent
		if (
			settingsIntent != null &&
			evidence.bestSettingsConfidence >= confidenceThreshold
		) {
			return if (settingsIntent == Intent.ABORT) {
				SettingsMenuIntentRoute.Abort
			} else {
				SettingsMenuIntentRoute.LocalSetting(
					intent = settingsIntent,
					confidence = evidence.bestSettingsConfidence
				)
			}
		}

		return SettingsMenuIntentRoute.Unresolved
	}

	fun evidenceFrom(intentResult: IntentResult): SettingsMenuIntentEvidence {
		val bestSettingsIntent = concreteSettingsIntents.maxByOrNull {
			intentResult.probabilityFor(it)
		}
		return SettingsMenuIntentEvidence(
			topIntent = intentResult.intent,
			topConfidence = intentResult.confidence,
			bestSettingsIntent = bestSettingsIntent,
			bestSettingsConfidence = bestSettingsIntent?.let {
				intentResult.probabilityFor(it)
			} ?: 0f
		)
	}

	fun isExternalIntent(intent: Intent): Boolean = intent in externalIntents
}
