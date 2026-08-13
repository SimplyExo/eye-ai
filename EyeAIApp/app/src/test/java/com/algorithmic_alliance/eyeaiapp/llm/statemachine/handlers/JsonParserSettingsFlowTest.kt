package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class JsonParserSettingsFlowTest {
	private val parser = JsonParser()

	@Test
	fun directContextPreservesOriginalVoskTextExactly() {
		val originalText = "  Setze bitte die Frequenz auf ungefähr \"700 Hertz\".\n"
		val context = parser.createSettingsContext(
			SettingIntent.FREQUENCY,
			SettingsFlow.DIRECT,
			originalText
		)

		assertEquals(SettingIntent.FREQUENCY, parser.parseSettingIntent(context))
		assertEquals(SettingsFlow.DIRECT, parser.parseSettingsFlow(context))
		assertEquals(originalText, parser.parseSettingsOriginalText(context))
	}

	@Test
	fun guidedContextDoesNotBecomeDirect() {
		val context = parser.createSettingsContext(SettingIntent.FREQUENCY)

		assertEquals(SettingsFlow.GUIDED, parser.parseSettingsFlow(context))
		assertEquals(null, parser.parseSettingsOriginalText(context))
	}

	@Test
	fun completeExtractionRequiresExpectedSettingField() {
		assertTrue(
			parser.hasExpectedSettingChange(
				"""{"settings_parameter_complete":true,"changed_settings":[{"frequency":700}]}""",
				SettingIntent.FREQUENCY
			)
		)
		assertFalse(
			parser.hasExpectedSettingChange(
				"""{"settings_parameter_complete":true,"changed_settings":[{"bps":4}]}""",
				SettingIntent.FREQUENCY
			)
		)
	}

	@Test
	fun emptyExtractionTriggersTargetedFollowUpPath() {
		assertFalse(
			parser.hasExpectedSettingChange(
				"""{"settings_parameter_complete":false,"changed_settings":[]}""",
				SettingIntent.FREQUENCY
			)
		)
		assertFalse(
			parser.hasExpectedSettingChange(
				"""{"setting_intent":"frequency"}""",
				SettingIntent.FREQUENCY
			)
		)
	}

	@Test
	fun directMetadataSurvivesGeminiExtraction() {
		val originalText = "Stell die Stimme auf männlich."
		val context = parser.createSettingsContext(
			SettingIntent.VOICE,
			SettingsFlow.DIRECT,
			originalText
		)
		val extracted = parser.carrySettingsContext(
			"""{"settings_parameter_complete":true,"changed_settings":[{"voice":1}]}""",
			context
		)

		assertTrue(parser.hasExpectedSettingChange(extracted, SettingIntent.VOICE))
		assertEquals(SettingsFlow.DIRECT, parser.parseSettingsFlow(extracted))
		assertEquals(originalText, parser.parseSettingsOriginalText(extracted))
		assertEquals(
			"Verstanden. Soll die Assistentenstimme nun männlich sein?",
			parser.createConfirmationQuestion(extracted)
		)
	}

	@Test
	fun invalidVoiceValueCannotReachConfirmation() {
		assertFalse(
			parser.hasExpectedSettingChange(
				"""{"settings_parameter_complete":true,"changed_settings":[{"voice":7}]}""",
				SettingIntent.VOICE
			)
		)
	}

	@Test
	fun approvalAndFullSettingsAbortRemainDistinct() {
		assertEquals(true, parser.parseApproval("""{"approval":1}"""))
		assertEquals(false, parser.parseApproval("""{"approval":0}"""))
		assertEquals(null, parser.parseApproval("{}"))
		assertFalse(parser.isSettingsFlowAbort("""{"approval":0}"""))
		assertTrue(
			parser.isSettingsFlowAbort(
				"""{"approval":0,"abort_settings_flow":true}"""
			)
		)
	}
}
