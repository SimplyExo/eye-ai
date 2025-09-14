package com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

class StreamParser(private val onChunk: (String) -> Unit) {

	fun parseStreamChunk(rawJsonCandidate: String) {
		try {
			Log.d(EyeAIApp.APP_LOG_TAG, "Attempting to parse JSON chunk (len=${rawJsonCandidate.length})...")
			val trimmed = rawJsonCandidate.trim()
			val root = parseJsonRoot(trimmed, rawJsonCandidate)

			val foundTexts = LinkedHashSet<String>()
			extractTextFromJson(root, foundTexts)

			foundTexts.forEach { text ->
				if (text.length >= 2) {
					Log.d(EyeAIApp.APP_LOG_TAG, "Parsed text (normalized & deduped): '${text.take(200)}...'")
					onChunk(text)
				}
			}

			Log.d(EyeAIApp.APP_LOG_TAG, "Finished parseStreamChunk.")
		} catch (e: JSONException) {
			Log.w(EyeAIApp.APP_LOG_TAG, "Could not parse stream chunk as JSON: '${rawJsonCandidate.take(200)}...'", e)
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "Unexpected error while parsing stream chunk", e)
		}
	}

	private fun parseJsonRoot(trimmed: String, rawJsonCandidate: String): Any {
		return try {
			if (trimmed.startsWith("[")) JSONArray(trimmed) else JSONObject(trimmed)
		} catch (_: JSONException) {
			// Fallback parsing attempts
			tryExtractJsonObject(rawJsonCandidate)
				?: tryExtractJsonArray(rawJsonCandidate)
				?: throw JSONException("Could not coerce chunk to JSON: '${rawJsonCandidate.take(200)}...'")
		}
	}

	private fun tryExtractJsonObject(raw: String): JSONObject? {
		val firstObj = raw.indexOf('{')
		val lastObj = raw.lastIndexOf('}')
		return if (firstObj != -1 && lastObj != -1 && lastObj > firstObj) {
			JSONObject(raw.substring(firstObj, lastObj + 1))
		} else null
	}

	private fun tryExtractJsonArray(raw: String): JSONArray? {
		val firstArr = raw.indexOf('[')
		val lastArr = raw.lastIndexOf(']')
		return if (firstArr != -1 && lastArr != -1 && lastArr > firstArr) {
			JSONArray(raw.substring(firstArr, lastArr + 1))
		} else null
	}

	private fun extractTextFromJson(node: Any?, foundTexts: MutableSet<String>) {
		when (node) {
			is JSONObject -> {
				processCandidates(node, foundTexts)
				processParts(node, foundTexts)
				processAllKeys(node, foundTexts)
			}
			is JSONArray -> {
				for (i in 0 until node.length()) {
					extractTextFromJson(node.opt(i), foundTexts)
				}
			}
		}
	}

	private fun processCandidates(node: JSONObject, foundTexts: MutableSet<String>) {
		val candidates = node.optJSONArray("candidates") ?: return
		for (i in 0 until candidates.length()) {
			val candidate = candidates.opt(i) as? JSONObject ?: continue
			processCandidate(candidate, foundTexts)
		}
	}

	private fun processCandidate(candidate: JSONObject, foundTexts: MutableSet<String>) {
		val content = candidate.optJSONObject("content") ?: return

		// Extract from parts
		val parts = content.optJSONArray("parts")
		if (parts != null) {
			for (i in 0 until parts.length()) {
				val part = parts.optJSONObject(i)
				val rawText = if (part != null) {
					part.optString("text", "")
				} else {
					parts.optString(i, "")
				}
				val normalizedText = normalizeText(rawText)
				if (normalizedText.isNotEmpty()) {
					foundTexts.add(normalizedText)
				}
			}
		}

		// Extract direct text from content
		val directText = normalizeText(content.optString("text", ""))
		if (directText.isNotEmpty()) {
			foundTexts.add(directText)
		}

		// Extract text from candidate itself
		val candidateText = normalizeText(candidate.optString("text", ""))
		if (candidateText.isNotEmpty()) {
			foundTexts.add(candidateText)
		}
	}

	private fun processParts(node: JSONObject, foundTexts: MutableSet<String>) {
		val parts = node.optJSONArray("parts") ?: return
		for (i in 0 until parts.length()) {
			val part = parts.optJSONObject(i)
			val rawText = if (part != null) {
				part.optString("text", "")
			} else {
				parts.optString(i, "")
			}
			val normalizedText = normalizeText(rawText)
			if (normalizedText.isNotEmpty()) {
				foundTexts.add(normalizedText)
			}
		}
	}

	private fun processAllKeys(node: JSONObject, foundTexts: MutableSet<String>) {
		val keys = node.keys()
		while (keys.hasNext()) {
			val key = keys.next()
			extractTextFromJson(node.opt(key), foundTexts)
		}
	}

	private fun normalizeText(text: String): String {
		return text.replace("\r", " ")
			.replace("\n", " ")
			.replace(Regex("\\s+"), " ")
			.trim()
	}
}
