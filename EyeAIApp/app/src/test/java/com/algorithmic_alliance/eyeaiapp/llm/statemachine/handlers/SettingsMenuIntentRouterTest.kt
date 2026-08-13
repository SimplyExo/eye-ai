package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.llm.LLM
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
			Intent.MEASURE_DISTANCE,
			Intent.REDIRECT_TO_LLM
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
	fun settingsCandidateBelowExistingThresholdUsesGeminiFallback() {
		val route = SettingsMenuIntentRouter.route(
			SettingsMenuIntentEvidence(
				topIntent = Intent.SET_FREQUENCY,
				topConfidence = 0.59f,
				bestSettingsIntent = Intent.SET_FREQUENCY,
				bestSettingsConfidence = 0.59f
			),
			threshold
		)

		assertTrue(route is SettingsMenuIntentRoute.GeminiFallback)
		assertEquals(
			0.59f,
			(route as SettingsMenuIntentRoute.GeminiFallback).bestSettingsConfidence
		)
	}

	@Test
	fun currentSignalrateModelEvidenceUsesExistingGeminiFallback() {
		val route = SettingsMenuIntentRouter.route(
			SettingsMenuIntentEvidence(
				topIntent = Intent.OPEN_SETTINGS,
				topConfidence = 0.3913814f,
				bestSettingsIntent = Intent.ABORT,
				bestSettingsConfidence = 0.3526614f
			),
			threshold
		)

		assertTrue(route is SettingsMenuIntentRoute.GeminiFallback)
		val fallbackLlm = object : LLM {
			override fun generate(command: String, structured: Boolean): String = ""
		}
		assertTrue(
			fallbackLlm.buildSettingsMenuPrompt("Signalrate.")
				.contains("Signalrate")
		)
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
