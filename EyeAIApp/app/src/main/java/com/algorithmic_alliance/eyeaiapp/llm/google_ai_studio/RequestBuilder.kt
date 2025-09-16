package com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio

import com.algorithmic_alliance.eyeaiapp.llm.LLM
import org.json.JSONArray
import org.json.JSONObject

object RequestBuilder {

	fun createRequestBody(prompt: String, structured: Boolean): JSONObject {
		val baseRequest = createBaseRequest(prompt)

		return if (structured) {
			baseRequest.put("generationConfig", createStructuredConfig())
		} else {
			baseRequest.put("generationConfig", JSONObject())
		}
	}

	private fun createBaseRequest(prompt: String): JSONObject {
		return JSONObject().apply {
			put("systemInstruction", createSystemInstruction())
			put("contents", createContents(prompt))
		}
	}

	private fun createSystemInstruction(): JSONObject {
		return JSONObject().apply {
			put("parts", JSONArray().apply {
				put(JSONObject().apply {
					put("text", LLM.SYSTEM_PROMPT)
				})
			})
		}
	}

	private fun createContents(prompt: String): JSONArray {
		return JSONArray().apply {
			put(JSONObject().apply {
				put("parts", JSONArray().apply {
					put(JSONObject().apply {
						put("text", prompt)
					})
				})
			})
		}
	}

	private fun createStructuredConfig(): JSONObject {
		return JSONObject().apply {
			put("response_mime_type", "application/json")
			put("response_schema", createSchema())
		}
	}

	private fun createSchema(): JSONObject {
		return JSONObject().apply {
			put("type", "OBJECT")
			put("properties", createSchemaProperties())
			put("required", createRequiredFields())
		}
	}

	private fun createSchemaProperties(): JSONObject {
		return JSONObject().apply {
			put("interaction_text", JSONObject().apply {
				put("type", "STRING")
			})
			put("object_query", JSONObject().apply {
				put("type", "STRING")
				put("description", "Das spezifische Objekt, nach dem der bei der Objekterkennung fragt Nutzer fragt. Z.B. 'Stuhl' oder 'Tisch'. IMMER SETZEN WENN OBJEKTERKENNUNG = TRUE!!!")
			})
			put("setting_intent", createSettingIntentProperty())
			put("requested_functions", createRequestedFunctionsProperty())
			put("changed_settings", createChangedSettingsProperty())
			put("approval", JSONObject().apply {
				put("type", "NUMBER")
			})
		}
	}

	private fun createSettingIntentProperty(): JSONObject {
		return JSONObject().apply {
			put("type", "STRING")
			put("enum", JSONArray().apply {
				put("tts_speed")
				put("voice")
				put("leave")
				put("none")
			})
		}
	}

	private fun createRequestedFunctionsProperty(): JSONObject {
		return JSONObject().apply {
			put("type", "OBJECT")
			put("properties", JSONObject().apply {
				put("einstellungen", JSONObject().apply { put("type", "BOOLEAN") })
				put("texterkennung", JSONObject().apply { put("type", "BOOLEAN") })
				put("objekterkennung", JSONObject().apply { put("type", "BOOLEAN") })
			})
		}
	}

	private fun createChangedSettingsProperty(): JSONObject {
		return JSONObject().apply {
			put("type", "ARRAY")
			put("items", JSONObject().apply {
				put("type", "OBJECT")
				put("properties", JSONObject().apply {
					put("tts_speed", JSONObject().apply { put("type", "NUMBER") })
					put("voice", JSONObject().apply { put("type", "NUMBER") })
					put("leave", JSONObject().apply { put("type", "BOOLEAN") })
				})
			})
		}
	}

	private fun createRequiredFields(): JSONArray {
		return JSONArray().apply {
			put("requested_functions")
			put("object_query")
			put("interaction_text")
			put("setting_intent")
		}
	}
}
