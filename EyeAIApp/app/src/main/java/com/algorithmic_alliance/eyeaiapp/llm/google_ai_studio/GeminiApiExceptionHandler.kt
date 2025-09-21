package com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp

class GeminiApiExceptionHandler(
	val errorCode: Int,
	val userMessage: String,
	val technicalMessage: String
) : Exception("Error $errorCode: $technicalMessage")

object GeminiErrorHandler {

	fun handleHttpError(responseCode: Int, errorResponse: String): GeminiApiExceptionHandler {
		Log.e(EyeAIApp.APP_LOG_TAG, "Gemini API Error $responseCode: ${errorResponse.take(1000)}")

		val userMessage = when (responseCode) {
			400 -> "Error 400: Es ist ein Fehler bei den Anfrage aufgetreten."
			403 -> "Error 403: Ihr API-Key zur Gemini API ist nicht zu Anfragen berechtigt."
			404 -> "Error 404: Die angeforderte Ressource wurde nicht gefunden."
			429 -> "Error 429: Das Ratelimit wurde überschritten, versuchen Sie es später nochmal."
			500 -> "Error 500: Es ist ein Fehler bei Google aufgetreten."
			503 -> "Error 503: Der Dienst ist überlastet oder vorübergehend nicht erreichbar."
			504 -> "Error 504: Ihre Anfrage dauert zu lange und kann derzeit nicht verarbeitet werden."
			else -> "Error $responseCode: Ein unbekannter Fehler ist aufgetreten."
		}

		return GeminiApiExceptionHandler(
			errorCode = responseCode,
			userMessage = userMessage,
			technicalMessage = errorResponse.ifBlank { "No error details provided" }
		)
	}

	//might be useful later-on when dealing with the Errors instead of redirecting it to the user
	@Suppress("unused")
	fun getErrorCategory(errorCode: Int): ErrorCategory {
		return when (errorCode) {
			400 -> ErrorCategory.CLIENT_ERROR
			403 -> ErrorCategory.AUTHENTICATION_ERROR
			404 -> ErrorCategory.NOT_FOUND_ERROR
			429 -> ErrorCategory.RATE_LIMIT_ERROR
			in 500..599 -> ErrorCategory.SERVER_ERROR
			else -> ErrorCategory.UNKNOWN_ERROR
		}
	}

	enum class ErrorCategory {
		CLIENT_ERROR,
		AUTHENTICATION_ERROR,
		NOT_FOUND_ERROR,
		RATE_LIMIT_ERROR,
		SERVER_ERROR,
		UNKNOWN_ERROR
	}
}
