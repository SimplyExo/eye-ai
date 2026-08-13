package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

class JsonParser {
	private companion object {
		const val SETTINGS_FLOW_KEY = "settings_flow"
		const val SETTINGS_ORIGINAL_TEXT_KEY = "settings_original_text"
		const val SETTINGS_PARAMETER_COMPLETE_KEY = "settings_parameter_complete"
	}

	fun parseRequestedFunction(jsonString: String): RequestedFunction {
		return try {
			val requestedFunctions = JSONObject(jsonString).optJSONObject("requested_functions")
			when {
				requestedFunctions?.optBoolean("texterkennung", false) == true -> RequestedFunction.TEXT_RECOGNITION
				requestedFunctions?.optBoolean("einstellungen", false) == true -> RequestedFunction.SETTINGS
				requestedFunctions?.optBoolean("objekterkennung", false) == true -> RequestedFunction.OBJECT_DETECTION
				else -> RequestedFunction.NONE
			}
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseRequestedFunction", e)
			RequestedFunction.NONE
		}
	}

	fun parseObjectQuery(jsonString: String): String? {
		return try {
			val query = JSONObject(jsonString).optString("object_query", null)
			if (query.isNullOrBlank()) null else query.trim()
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseObjectQuery", e)
			null
		}
	}

	fun parseSettingIntent(jsonString: String): SettingIntent {
		return try {
			SettingIntent.fromWireValue(
				JSONObject(jsonString).optString("setting_intent", SettingIntent.NONE.wireValue)
			)
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseSettingIntent", e)
			SettingIntent.NONE
		}
	}

	fun createSettingsContext(
		settingIntent: SettingIntent,
		flow: SettingsFlow = SettingsFlow.GUIDED,
		originalText: String? = null
	): String = JSONObject().apply {
		put("requested_function", "settings")
		put("setting_intent", settingIntent.wireValue)
		if (flow == SettingsFlow.DIRECT) {
			put(SETTINGS_FLOW_KEY, flow.wireValue)
			originalText?.let { put(SETTINGS_ORIGINAL_TEXT_KEY, it) }
		}
	}.toString()

	fun parseSettingsFlow(jsonString: String?): SettingsFlow {
		if (jsonString == null) return SettingsFlow.GUIDED
		return try {
			SettingsFlow.fromWireValue(
				JSONObject(jsonString).optString(SETTINGS_FLOW_KEY, SettingsFlow.GUIDED.wireValue)
			)
		} catch (_: JSONException) {
			SettingsFlow.GUIDED
		}
	}

	fun parseSettingsOriginalText(jsonString: String?): String? {
		if (jsonString == null) return null
		return try {
			JSONObject(jsonString)
				.optString(SETTINGS_ORIGINAL_TEXT_KEY, "")
				.takeIf { it.isNotEmpty() }
		} catch (_: JSONException) {
			null
		}
	}

	/** Carries local flow metadata forward without asking Gemini to reproduce it. */
	fun carrySettingsContext(jsonResponse: String, currentJson: String?): String {
		if (parseSettingsFlow(currentJson) != SettingsFlow.DIRECT) return jsonResponse

		return try {
			JSONObject(jsonResponse).apply {
				put(SETTINGS_FLOW_KEY, SettingsFlow.DIRECT.wireValue)
				parseSettingsOriginalText(currentJson)?.let {
					put(SETTINGS_ORIGINAL_TEXT_KEY, it)
				}
			}.toString()
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Could not carry direct-settings context", e)
			jsonResponse
		}
	}

	/**
	 * Checks only Gemini's structured response, never the user's natural-language
	 * input. A missing expected field means that a targeted follow-up is needed.
	 */
	fun hasExpectedSettingChange(jsonString: String, settingIntent: SettingIntent): Boolean {
		return normalizedExpectedSettingChange(jsonString, settingIntent) != null
	}

	/** Keeps exactly one validated change so a confirmation can apply it only once. */
	fun normalizedExpectedSettingChange(
		jsonString: String,
		settingIntent: SettingIntent
	): String? {
		val expectedKey = settingIntent.changedSettingKey ?: return null
		return try {
			val root = JSONObject(jsonString)
			if (!root.optBoolean(SETTINGS_PARAMETER_COMPLETE_KEY, false)) return null
			val changes = root.optJSONArray("changed_settings") ?: return null
			for (index in 0 until changes.length()) {
				val change = changes.optJSONObject(index) ?: continue
				val value = change.opt(expectedKey)
				if (isUsableSettingValue(settingIntent, value)) {
					return root.apply {
						put("changed_settings", JSONArray().apply {
							put(JSONObject().apply { put(expectedKey, value) })
						})
					}.toString()
				}
			}
			null
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Could not validate extracted setting", e)
			null
		}
	}

	private fun isUsableSettingValue(settingIntent: SettingIntent, value: Any?): Boolean {
		val number = value as? Number ?: return false
		val finiteValue = number.toDouble()
		if (!finiteValue.isFinite()) return false

		return when (settingIntent) {
			SettingIntent.TTS_SPEED -> finiteValue > 0.0
			SettingIntent.VOICE -> finiteValue == 0.0 || finiteValue == 1.0
			SettingIntent.FREQUENCY, SettingIntent.BPS -> true
			SettingIntent.LEAVE, SettingIntent.NONE -> false
		}
	}

	fun parseInteractionText(jsonString: String): String? {
		return try {
			val obj = JSONObject(jsonString)
			val txt = obj.optString("interaction_text", "").trim()
			if (txt.isNotEmpty()) return txt


			val direct = obj.optString("text", "").trim()
			if (direct.isNotEmpty()) return direct
			null
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseInteractionText", e)
			null
		}
	}

	fun createConfirmationQuestion(jsonString: String): String {
		try {
			val changedSettings = JSONObject(jsonString).optJSONArray("changed_settings")
			if (changedSettings != null && changedSettings.length() > 0) {
				val firstChange = changedSettings.getJSONObject(0)
				when {
					firstChange.has("tts_speed") -> {
						val newSpeed = firstChange.getDouble("tts_speed")
						return "Verstanden. Soll ich die Sprachgeschwindigkeit auf $newSpeed setzen?"
					}
					firstChange.has("voice") -> {
						val voice = firstChange.getInt("voice")
						return if (voice == 1) "Verstanden. Soll die Assistentenstimme nun männlich sein?"
						else "Verstanden. Soll die Assistentenstimme nun weiblich sein?"
					}
					firstChange.has("frequency") -> {
						val frequency = firstChange.getInt("frequency")
						return "Verstanden. Soll ich die Audio-Frequenz auf $frequency Hz setzen?"
					}
					firstChange.has("bps") -> {
						val bps = firstChange.getInt("bps")
						return "Verstanden. Soll ich die BPS auf $bps setzen?"
					}
					firstChange.has("leave") -> return "Möchten Sie die Einstellungen wirklich verlassen?"
				}
			}
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in createConfirmationQuestion", e)
		}
		return "Soll ich die angeforderte Änderung durchführen?"
	}

	fun isApproved(jsonString: String): Boolean {
		return try {
			JSONObject(jsonString).optInt("approval", 0) == 1
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in isApproved", e)
			false
		}
	}

	fun parseExtractedObject(jsonString: String): String? {
		return try {
			val jsonObject = JSONObject(jsonString)
			jsonObject.optString("extracted_object", null).takeIf { !it.isNullOrBlank() }
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseExtractedObject", e)
			null
		}
	}

	fun isLeaveRequest(jsonString: String): Boolean {
		return try {
			val json = JSONObject(jsonString)
			if (json.has("changed_settings")) {
				val settings = json.getJSONArray("changed_settings")
				for (i in 0 until settings.length()) {
					val setting = settings.getJSONObject(i)
					if (setting.has("leave") && setting.getBoolean("leave")) {
						return true
					}
				}
			}
			false
		} catch (e: Exception) {
			false
		}
	}

}
