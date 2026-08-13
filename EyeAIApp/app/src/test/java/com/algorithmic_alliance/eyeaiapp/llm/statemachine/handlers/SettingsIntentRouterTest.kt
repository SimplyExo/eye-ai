package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.nlp.Intent
import com.algorithmic_alliance.eyeaiapp.nlp.IntentResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class SettingsIntentRouterTest {
	@Test
	fun openSettingsKeepsGuidedMenuRoute() {
		val result = intentResult(Intent.OPEN_SETTINGS, "Öffne die Einstellungen.")

		assertSame(SettingsIntentRoute.GuidedMenu, SettingsIntentRouter.route(result))
	}

	@Test
	fun concreteSettingsIntentsRouteDirectlyWithUnchangedOriginalText() {
		val examples = listOf(
			Triple(Intent.SET_FREQUENCY, SettingIntent.FREQUENCY, "Setze bitte die Frequenz auf ungefähr 700 Hertz."),
			Triple(Intent.CHANGE_SPEAKER, SettingIntent.VOICE, "Stell die Stimme auf männlich."),
			Triple(Intent.CHANGE_SPEECH_SPEED, SettingIntent.TTS_SPEED, "Sprich schneller."),
			Triple(Intent.SET_BPS, SettingIntent.BPS, "Setze die Signalrate auf 4 BPS.")
		)

		examples.forEach { (intent, expectedSettingIntent, originalText) ->
			val intentResult = intentResult(intent, originalText)
			val route = SettingsIntentRouter.route(intentResult)

			assertTrue(route is SettingsIntentRoute.Direct)
			route as SettingsIntentRoute.Direct
			assertEquals(expectedSettingIntent, route.settingIntent)
			assertSame(intentResult, route.intentResult)
			assertEquals(originalText, route.intentResult.originalText)
		}
	}

	@Test
	fun incompleteDirectCommandsStillUseTheirTargetedDirectRoute() {
		val examples = listOf(
			Intent.SET_FREQUENCY to "Ändere die Frequenz.",
			Intent.CHANGE_SPEAKER to "Ändere die Stimme.",
			Intent.CHANGE_SPEECH_SPEED to "Ändere die Geschwindigkeit.",
			Intent.SET_BPS to "Ändere die Signalrate."
		)

		examples.forEach { (intent, originalText) ->
			assertTrue(SettingsIntentRouter.route(intentResult(intent, originalText)) is SettingsIntentRoute.Direct)
		}
	}

	@Test
	fun nonSettingsIntentsRemainOutsideSettingsRouting() {
		val result = intentResult(Intent.OBJECT_DETECTION, "Was ist vor mir?")

		assertSame(SettingsIntentRoute.NotSettings, SettingsIntentRouter.route(result))
	}

	@Test
	fun cancellationReturnsDirectFlowToIdleAndGuidedFlowToMenu() {
		assertEquals(SettingsCancellationDestination.IDLE, SettingsFlow.DIRECT.cancellationDestination())
		assertEquals(SettingsCancellationDestination.GUIDED_MENU, SettingsFlow.GUIDED.cancellationDestination())
	}

	@Test
	fun abortPhraseIsRecognizedByExistingKeywordPath() {
		assertTrue(SettingsExitCommandDetector.matches("Abbrechen."))
		assertTrue(SettingsExitCommandDetector.matches("Bitte stopp den Vorgang"))
		assertEquals(false, SettingsExitCommandDetector.matches("Ja, bitte ausführen."))
	}

	private fun intentResult(intent: Intent, originalText: String): IntentResult {
		val probabilities = FloatArray(Intent.CLASS_ORDER.size)
		probabilities[intent.ordinal] = 1f
		return IntentResult(intent, 1f, originalText, probabilities)
	}
}
