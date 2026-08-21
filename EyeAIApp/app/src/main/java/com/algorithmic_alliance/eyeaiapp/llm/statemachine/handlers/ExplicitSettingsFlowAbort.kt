package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import java.text.Normalizer
import java.util.Locale

/** Preserves the pre-existing whole-dialog abort control outside the 3-label model. */
object ExplicitSettingsFlowAbort {
	private val phrases = setOf(
		"abbrechen",
		"abbruch",
		"stopp",
		"stop",
		"alles abbrechen",
		"dialog abbrechen",
		"einstellungsdialog abbrechen"
	)

	fun matches(input: String): Boolean = normalize(input) in phrases

	private fun normalize(input: String): String =
		Normalizer.normalize(input, Normalizer.Form.NFKC)
			.lowercase(Locale.ROOT)
			.replace(Regex("[^\\p{L}\\p{N}_]+"), " ")
			.trim()
			.replace(Regex(" +"), " ")
}
