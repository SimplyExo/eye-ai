package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.llm.LLM
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GeminiSettingsProcessorTest {
	private val parser = JsonParser()
	private val snapshot = CurrentSettingsSnapshot(
		speechRate = 1.0f,
		voice = 0,
		frequency = 500,
		bps = 2
	)

	@Test
	fun completeDirectFrequencyUsesOriginalTextInExactlyOneGeminiRequest() = runBlocking {
		val originalText = "Setze bitte die Frequenz auf ungefähr 700 Hertz."
		val context = parser.createSettingsContext(
			SettingIntent.FREQUENCY,
			SettingsFlow.DIRECT,
			originalText
		)
		var requestCount = 0
		var capturedPrompt = ""
		val traces = mutableListOf<String>()
		val extractor = GeminiSettingsExtractor(
			jsonParser = parser,
			trace = traces::add
		) { prompt, structured ->
			requestCount++
			capturedPrompt = prompt
			assertTrue(structured)
			"""{"settings_parameter_complete":true,"changed_settings":[{"frequency":700}]}"""
		}

		val result = extractor.extract(SettingIntent.FREQUENCY, originalText, context, snapshot)

		assertTrue(result is SettingsExtractionResult.Complete)
		result as SettingsExtractionResult.Complete
		assertEquals(1, requestCount)
		assertTrue(capturedPrompt.contains("'$originalText'"))
		assertEquals(SettingsFlow.DIRECT, parser.parseSettingsFlow(result.json))
		assertEquals(originalText, parser.parseSettingsOriginalText(result.json))
		assertEquals("Verstanden. Soll ich die Audio-Frequenz auf 700 Hz setzen?", result.confirmationQuestion)
		assertTrue(
			traces.any {
				it.contains("[DecisionTrace][Gemini API][EVALUATE]") &&
					it.contains("role=SETTINGS_PARAMETER_EXTRACTION")
			}
		)
		assertTrue(traces.any { it.contains("outcome=COMPLETE") })
	}

	@Test
	fun incompleteCommandsAskTargetedQuestionWithoutOpeningGeneralMenu() = runBlocking {
		val examples = listOf(
			Triple(SettingIntent.FREQUENCY, "Ändere die Frequenz.", LLM.SNIPPET_FREQUENCY),
			Triple(SettingIntent.VOICE, "Ändere die Stimme.", LLM.SNIPPET_VOICE),
			Triple(SettingIntent.TTS_SPEED, "Ändere die Geschwindigkeit.", LLM.SNIPPET_TTS_SPEED),
			Triple(SettingIntent.BPS, "Ändere die Signalrate.", LLM.SNIPPET_BPS)
		)

		examples.forEach { (settingIntent, originalText, expectedQuestion) ->
			var requestCount = 0
			val extractor = GeminiSettingsExtractor(parser) { _, _ ->
				requestCount++
				"""{"settings_parameter_complete":false,"changed_settings":[]}"""
			}
			val result = extractor.extract(
				settingIntent,
				originalText,
				parser.createSettingsContext(
					settingIntent,
					SettingsFlow.DIRECT,
					originalText
				),
				snapshot
			)

			assertEquals(1, requestCount)
			assertTrue(result is SettingsExtractionResult.MissingValue)
			result as SettingsExtractionResult.MissingValue
			assertEquals(expectedQuestion, result.targetedQuestion)
		}
	}

	@Test
	fun followUpExtractionIncludesOriginalCommandAndLatestAnswer() = runBlocking {
		val originalText = "Ändere die Frequenz."
		val followUp = "700 Hertz"
		val context = parser.createSettingsContext(
			SettingIntent.FREQUENCY,
			SettingsFlow.DIRECT,
			originalText
		)
		var capturedPrompt = ""
		val extractor = GeminiSettingsExtractor(parser) { prompt, _ ->
			capturedPrompt = prompt
			"""{"settings_parameter_complete":true,"changed_settings":[{"frequency":700}]}"""
		}

		val result = extractor.extract(SettingIntent.FREQUENCY, followUp, context, snapshot)

		assertTrue(result is SettingsExtractionResult.Complete)
		assertTrue(capturedPrompt.contains("'$originalText'"))
		assertTrue(capturedPrompt.contains("'$followUp'"))
	}

	@Test
	fun allConcreteSettingsUseExistingGeminiExtraction() = runBlocking {
		val examples = listOf(
			Triple(SettingIntent.VOICE, "Stell die Stimme auf männlich.", """{"settings_parameter_complete":true,"changed_settings":[{"voice":1}]}"""),
			Triple(SettingIntent.TTS_SPEED, "Sprich schneller.", """{"settings_parameter_complete":true,"changed_settings":[{"tts_speed":1.2}]}"""),
			Triple(SettingIntent.BPS, "Setze die Signalrate auf 4 BPS.", """{"settings_parameter_complete":true,"changed_settings":[{"bps":4}]}""")
		)

		examples.forEach { (settingIntent, originalText, response) ->
			var requestCount = 0
			val extractor = GeminiSettingsExtractor(parser) { _, _ ->
				requestCount++
				response
			}
			val context = parser.createSettingsContext(
				settingIntent,
				SettingsFlow.DIRECT,
				originalText
			)

			assertTrue(
				extractor.extract(settingIntent, originalText, context, snapshot) is
					SettingsExtractionResult.Complete
			)
			assertEquals(1, requestCount)
		}
	}

	@Test
	fun extractionNormalizesGeminiOutputToOneExpectedChange() = runBlocking {
		val extractor = GeminiSettingsExtractor(parser) { _, _ ->
			"""{"settings_parameter_complete":true,"changed_settings":[{"frequency":700,"bps":4},{"frequency":900}]}"""
		}

		val result = extractor.extract(
			SettingIntent.FREQUENCY,
			"Setze die Frequenz auf 700 Hertz.",
			parser.createSettingsContext(SettingIntent.FREQUENCY),
			snapshot
		) as SettingsExtractionResult.Complete
		val changes = JSONObject(result.json).getJSONArray("changed_settings")

		assertEquals(1, changes.length())
		assertEquals(setOf("frequency"), changes.getJSONObject(0).keys().asSequence().toSet())
		assertEquals(700, changes.getJSONObject(0).getInt("frequency"))
	}

	@Test
	fun approvedChangeGeneratesOnceAndAppliesExactlyOnce() = runBlocking {
		var requestCount = 0
		var applyCount = 0
		val traces = mutableListOf<String>()
		val confirmation = GeminiSettingsConfirmation(
			jsonParser = parser,
			trace = traces::add
		) { _, structured ->
			requestCount++
			assertTrue(structured)
			"""{"approval":1}"""
		}

		val result = confirmation.confirmAndApply(
			"Ja.",
			"""{"changed_settings":[{"frequency":700}]}"""
		) {
			applyCount++
			true
		}

		assertEquals(SettingsConfirmationResult.APPLIED, result)
		assertEquals(1, requestCount)
		assertEquals(1, applyCount)
		assertTrue(
			traces.any {
				it.contains("role=SETTINGS_CONFIRMATION") &&
					it.contains("outcome=APPROVED_AND_APPLIED")
			}
		)
	}

	@Test
	fun rejectedChangeDoesNotApply() = runBlocking {
		var requestCount = 0
		var applyCount = 0
		val confirmation = GeminiSettingsConfirmation(parser) { _, _ ->
			requestCount++
			"""{"approval":0}"""
		}

		val result = confirmation.confirmAndApply(
			"Nein.",
			"""{"changed_settings":[{"frequency":700}]}"""
		) {
			applyCount++
			true
		}

		assertEquals(SettingsConfirmationResult.REJECTED, result)
		assertEquals(1, requestCount)
		assertEquals(0, applyCount)
	}
}
