package com.algorithmic_alliance.eyeaiapp.llm.statemachine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GenericCancellationTest {
	@Test
	fun cancellationTextIsExactForSettingsRedirectAndGenericFlows() {
		val contexts = listOf("settings", "redirect", "generic")

		contexts.forEach { _ ->
			assertEquals(
				"Ich habe den Vorgang abgebrochen.",
				GenericCancellation.responseFor("abbrechen")
			)
		}
	}

	@Test
	fun commonCancellationVariantsUseTheSameResponse() {
		listOf("Abbrechen.", "ABBRUCH", "Stopp", "Brich den Vorgang ab").forEach { input ->
			assertTrue(GenericCancellation.matches(input))
			assertEquals(GenericCancellation.RESPONSE, GenericCancellation.responseFor(input))
		}
	}
}
