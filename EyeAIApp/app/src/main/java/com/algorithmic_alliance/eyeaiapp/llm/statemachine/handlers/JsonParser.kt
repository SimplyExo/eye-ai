package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import org.json.JSONException
import org.json.JSONObject

class JsonParser {

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
			if (query.isNullOrBlank()) null else query
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseObjectQuery", e)
			null
		}
	}

	fun parseSettingIntent(jsonString: String): SettingIntent {
		return try {
			when (JSONObject(jsonString).optString("setting_intent", "none")) {
				"tts_speed" -> SettingIntent.TTS_SPEED
				"voice" -> SettingIntent.VOICE
				"leave" -> SettingIntent.LEAVE
				else -> SettingIntent.NONE
			}
		} catch (e: JSONException) {
			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed in parseSettingIntent", e)
			SettingIntent.NONE
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
						return if (voice == 1) "Verstanden. Soll die Assistentenstimme nun weiblich sein?"
						else "Verstanden. Soll die Assistentenstimme nun männlich sein?"
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

}