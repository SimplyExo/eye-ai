package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.llm.statemachine.LocalInteractionMessages
import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import com.algorithmic_alliance.eyeaiapp.nlp.IntentResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class SettingsMenuIntentRouterTest {
	private val threshold = 0.6f

	@Test
	fun existingConcreteSettingsIntentsKeepLocalRouting() {
		val intents = listOf(
			Intent.SET_FREQUENCY,
			Intent.CHANGE_SPEAKER,
			Intent.SET_BPS,
			Intent.CHANGE_SPEECH_SPEED
		)

		intents.forEach { intent ->
			val route = SettingsMenuIntentRouter.route(oneHotResult(intent), threshold)
			assertTrue(route is SettingsMenuIntentRoute.LocalSetting)
			route as SettingsMenuIntentRoute.LocalSetting
			assertEquals(intent, route.intent)
			assertEquals(1f, route.confidence)
		}
	}

	@Test
	fun abortIsAnNlpRouteAndDoesNotNeedKeywordMatching() {
		assertSame(
			SettingsMenuIntentRoute.Abort,
			SettingsMenuIntentRouter.route(oneHotResult(Intent.ABORT), threshold)
		)
	}

	@Test
	fun allSupportedGlobalTop1IntentsRequestContextSwitch() {
		val globalIntents = listOf(
			Intent.TEXT_RECOGNITION,
			Intent.OBJECT_DETECTION,
			Intent.MEASURE_DISTANCE
		)

		globalIntents.forEach { intent ->
			val result = oneHotResult(intent)
			val route = SettingsMenuIntentRouter.route(result, threshold)
			assertTrue(route is SettingsMenuIntentRoute.ExternalIntent)
			route as SettingsMenuIntentRoute.ExternalIntent
			assertEquals(intent, route.intent)
			assertEquals(result.confidence, route.confidence)
		}
	}

	@Test
	fun redirectToLlmIsHandledAsLocalUnresolvedCommand() {
		val route = SettingsMenuIntentRouter.route(oneHotResult(Intent.REDIRECT_TO_LLM), threshold)

		assertSame(SettingsMenuIntentRoute.Unresolved, route)
		assertEquals(
			"Ich habe den Befehl nicht eindeutig verstanden. Bitte versuchen Sie es noch einmal.",
			LocalInteractionMessages.UNRESOLVED_COMMAND
		)
		assertTrue(!SettingsMenuIntentRouter.isExternalIntent(Intent.REDIRECT_TO_LLM))
	}

	@Test
	fun globalTop1CannotBeReplacedByWeakerSettingsCandidateAboveThreshold() {
		val route = SettingsMenuIntentRouter.route(
			SettingsMenuIntentEvidence(
				topIntent = Intent.TEXT_RECOGNITION,
				topConfidence = 0.94f,
				bestSettingsIntent = Intent.SET_FREQUENCY,
				bestSettingsConfidence = 0.63f
			),
			threshold
		)

		assertTrue(route is SettingsMenuIntentRoute.ExternalIntent)
		assertEquals(
			Intent.TEXT_RECOGNITION,
			(route as SettingsMenuIntentRoute.ExternalIntent).intent
		)
	}

	@Test
	fun strongSettingsIntentKeepsNormalSettingsRoute() {
		val route = SettingsMenuIntentRouter.route(
			SettingsMenuIntentEvidence(
				topIntent = Intent.SET_FREQUENCY,
				topConfidence = 0.90f,
				bestSettingsIntent = Intent.SET_FREQUENCY,
				bestSettingsConfidence = 0.90f
			),
			threshold
		)

		assertTrue(route is SettingsMenuIntentRoute.LocalSetting)
		assertEquals(
			Intent.SET_FREQUENCY,
			(route as SettingsMenuIntentRoute.LocalSetting).intent
		)
	}

	@Test
	fun settingsCandidateBelowThresholdIsUnresolved() {
		val route = SettingsMenuIntentRouter.route(
			SettingsMenuIntentEvidence(
				topIntent = Intent.SET_FREQUENCY,
				topConfidence = 0.59f,
				bestSettingsIntent = Intent.SET_FREQUENCY,
				bestSettingsConfidence = 0.59f
			),
			threshold
		)

		assertSame(SettingsMenuIntentRoute.Unresolved, route)
	}

	@Test
	fun ambiguousModelEvidenceIsUnresolved() {
		val route = SettingsMenuIntentRouter.route(
			SettingsMenuIntentEvidence(
				topIntent = Intent.OPEN_SETTINGS,
				topConfidence = 0.3913814f,
				bestSettingsIntent = Intent.ABORT,
				bestSettingsConfidence = 0.3526614f
			),
			threshold
		)

		assertSame(SettingsMenuIntentRoute.Unresolved, route)
	}

	@Test
	fun openSettingsInsideMenuDoesNotCreateNestedSettingsFlow() {
		assertSame(
			SettingsMenuIntentRoute.AlreadyInSettings,
			SettingsMenuIntentRouter.route(oneHotResult(Intent.OPEN_SETTINGS), threshold)
		)
	}

	private fun oneHotResult(intent: Intent): IntentResult {
		val probabilities = FloatArray(Intent.CLASS_ORDER.size)
		probabilities[intent.ordinal] = 1f
		return IntentResult(
			intent = intent,
			confidence = 1f,
			originalText = "test-$intent",
			probabilities = probabilities
		)
	}
}
