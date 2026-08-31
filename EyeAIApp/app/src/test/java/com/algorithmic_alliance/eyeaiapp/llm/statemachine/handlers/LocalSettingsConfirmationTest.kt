package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import com.algorithmic_alliance.eyeaiapp.confirmation.ConfirmationModelTestFixture
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalSettingsConfirmationTest {
	private val parser = JsonParser()
	private val confirmationModel by lazy(ConfirmationModelTestFixture::load)

	@Test
	fun approvedChangeIsLocalAndAppliesExactlyOnce() = runBlocking {
		var applyCount = 0
		val traces = mutableListOf<String>()
		val confirmation = LocalSettingsConfirmation(
			confirmationModelProvider = { confirmationModel },
			jsonParser = parser,
			trace = traces::add
		)

		val result = confirmation.confirmAndApply(
			"Ja.",
			"""{"changed_settings":[{"frequency":700}]}"""
		) {
			applyCount++
			true
		}

		assertEquals(SettingsConfirmationResult.APPLIED, result)
		assertEquals(1, applyCount)
		assertTrue(traces.any { it.contains("evaluator=LOCAL_CONFIRMATION_MODEL") })
		assertTrue(traces.all { it.contains("apiCalled=false") })
		assertTrue(traces.any { it.contains("decision=ACCEPT") && it.contains("confirmed=true") })
	}

	@Test
	fun unavailableVoiceIsNotReportedAsApplied() = runBlocking {
		val confirmation = LocalSettingsConfirmation({ confirmationModel }, parser)

		val result = confirmation.confirmAndApplyWithResult(
			"Ja.",
			"""{"changed_settings":[{"voice":1}]}"""
		) {
			SettingsApplyResult.NOT_APPLIED
		}

		assertEquals(SettingsConfirmationResult.NOT_APPLIED, result)
	}

	@Test
	fun rejectedChangeDoesNotApply() = runBlocking {
		var applyCount = 0
		val confirmation = LocalSettingsConfirmation({ confirmationModel }, parser)

		val result = confirmation.confirmAndApply(
			"Nein.",
			"""{"changed_settings":[{"frequency":700}]}"""
		) {
			applyCount++
			true
		}

		assertEquals(SettingsConfirmationResult.REJECTED, result)
		assertEquals(0, applyCount)
	}

	@Test
	fun unknownDoesNotApply() = runBlocking {
		var applyCount = 0
		val confirmation = LocalSettingsConfirmation({ confirmationModel }, parser)

		val result = confirmation.confirmAndApply(
			"Ich bin unsicher.",
			"""{"changed_settings":[{"frequency":700}]}"""
		) {
			applyCount++
			true
		}

		assertEquals(SettingsConfirmationResult.UNKNOWN, result)
		assertEquals(0, applyCount)
	}
}
