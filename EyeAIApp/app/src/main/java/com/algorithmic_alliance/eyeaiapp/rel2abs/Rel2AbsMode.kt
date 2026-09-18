package com.algorithmic_alliance.eyeaiapp.rel2abs

/** The explicit user-selected frozen REL2ABS calibration path. */
enum class Rel2AbsMode(val preferenceValue: String) {
	Z1("Z1"),
	S2("S2"),

	;

	companion object {
		fun fromPreference(value: String?): Rel2AbsMode =
			entries.firstOrNull { it.preferenceValue == value } ?: Z1
	}
}
