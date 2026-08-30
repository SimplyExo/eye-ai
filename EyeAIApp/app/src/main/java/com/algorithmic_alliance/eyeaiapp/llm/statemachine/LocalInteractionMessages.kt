package com.algorithmic_alliance.eyeaiapp.llm.statemachine

/** Local, deterministic responses shared by the interaction state machine. */
object LocalInteractionMessages {
	const val UNRESOLVED_COMMAND =
		"Ich habe den Befehl nicht eindeutig verstanden. Bitte versuchen Sie es noch einmal."

	const val SETTINGS_MENU = """Sehr gerne, ich kann Ihnen dabei helfen, die Einstellungen anzupassen. 

	Es besteht die Möglichkeit die Sprechgeschwindigkeit der Sprachausgabe anzupassen, die Stimme des Assitentenagenten zu ändern, die Tonhöhe der Distanztöne zu ändern,
	die Schläge pro Sekunde für die Distanzhinweistöne zu ändern, oder aber die Einstellungen zu verlassen. 

	Welche dieser Optionen möchten Sie wählen?"""
}
