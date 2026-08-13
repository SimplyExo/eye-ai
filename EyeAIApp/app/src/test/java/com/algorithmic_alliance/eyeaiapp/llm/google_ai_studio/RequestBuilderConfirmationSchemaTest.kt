package com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RequestBuilderConfirmationSchemaTest {
	@Test
	fun structuredGeminiSchemaSupportsApprovalAndExplicitSettingsAbort() {
		val request = RequestBuilder.createRequestBody("Bestätigen?", structured = true)
		val properties = request
			.getJSONObject("generationConfig")
			.getJSONObject("response_schema")
			.getJSONObject("properties")

		assertEquals("NUMBER", properties.getJSONObject("approval").getString("type"))
		assertEquals(
			"BOOLEAN",
			properties.getJSONObject("abort_settings_flow").getString("type")
		)
		assertTrue(
			properties.getJSONObject("abort_settings_flow")
				.getString("description")
				.contains("gesamten Einstellungsdialogs")
		)
	}
}
