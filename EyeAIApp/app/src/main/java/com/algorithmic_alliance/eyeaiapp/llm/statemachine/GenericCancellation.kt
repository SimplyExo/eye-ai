package com.algorithmic_alliance.eyeaiapp.llm.statemachine

import java.text.Normalizer
import java.util.Locale

/** One context-independent cancellation contract for all conversational flows. */
object GenericCancellation {
	const val RESPONSE = "Ich habe den Vorgang abgebrochen."

	private val phrases = setOf(
		"abbrechen",
		"abbruch",
		"stopp",
		"stop",
		"alles abbrechen",
		"dialog abbrechen",
		"brich ab",
		"brich den vorgang ab",
		"vorgang abbrechen"
	)

	fun matches(input: String): Boolean = normalize(input) in phrases

	fun responseFor(input: String): String? = RESPONSE.takeIf { matches(input) }

	private fun normalize(input: String): String =
		Normalizer.normalize(input, Normalizer.Form.NFKC)
			.lowercase(Locale.ROOT)
			.replace(Regex("[^\\p{L}\\p{N}_]+"), " ")
			.trim()
			.replace(Regex(" +"), " ")
}
