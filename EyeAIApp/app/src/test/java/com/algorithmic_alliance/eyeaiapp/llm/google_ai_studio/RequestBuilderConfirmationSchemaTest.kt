package com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class RequestBuilderConfirmationSchemaTest {
	@Test
	fun structuredGeminiSchemaUsesApprovalWithoutSettingsSpecificCancellationField() {
		val request = RequestBuilder.createRequestBody("Bestätigen?", structured = true)
		val properties = request
			.getJSONObject("generationConfig")
			.getJSONObject("response_schema")
			.getJSONObject("properties")

		assertEquals("NUMBER", properties.getJSONObject("approval").getString("type"))
		assertFalse(properties.has("abort_settings_flow"))
	}
}
