package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects

import java.util.Locale


object DistanceQueryExtractor {

	//Finds the longest known German detector label mentioned in [input]. The
	 // longest match keeps multi-word labels such as "Hot Dog".

	fun extract(input: String): String? {
		val normalizedInput = input.lowercase(Locale.GERMAN)
		return TranslateEnglishToGerman.getKnownGermanLabels()
			.asSequence()
			.sortedByDescending { it.length }
			.firstOrNull { label -> normalizedInput.contains(label.lowercase(Locale.GERMAN)) }
	}
}
