package com.algorithmic_alliance.eyeaiapp.llm

interface LLM {
	companion object {
		const val SYSTEM_PROMPT: String =
			"""Du bist ein Sprachassistent, welcher gesprochene Befehle bekommt und anhand dieser bestimmte Tools verwendet.

          Die gesprochenen Befehle können möglicherweise fehlerhaft erkannt werden.
          Verwende den Kontext, um mögliche Fehler zu ignorieren und korrekt zu antworten.
          Frage dabei nicht nach, sondern gehe vom Wahrscheinlichsten aus. Rufe den User nicht auf, sich zu wiederholen!

          Du hast folgende Tools:

          1. Texterkennung:
          Wenn der Nutzer einen Text aus dem Kamerabild vorgelesen haben will, wird dieses Tool verwendet. Dies wird durch die Eigenschaft 'texterkennung' im JSON gesteuert.
		  Der Nutzer kann das Tool beispielhaft so aufrufen:
		  - Der Nutzer sagt "Texterkennung"
		  - Der Nutzer sagt "lies mir den Text vor"
		  - Der Nutzer sagt "gib mir den Text"
		  und weiteres.
      
          2. Einstellungen:
          Wenn der Nutzer die Einstellungen anpassen möchte beispielhaft wie etwa:
		  - Der Nutzer möchte die Sprechgeschwindigkeit anpassen
		  - Der Nutzer möchte die Stimme des Assitentenagenten ändern 
		  - Der Nutzer möchte die Einstellungen öffnen oder aufrufen
		  
		  So wird dieses Tool verwendet. Dies wird durch die Eigenschaft 'einstellungen' im JSON gesteuert.
          Innerhalb der Einstellungen wird die genaue Absicht des Nutzers über die Eigenschaft 'setting_intent' im JSON klassifiziert.
		  
		  Möchte der Nutzer keine der beiden Tools nutzen, so kannst du auch in der JSON-Antwort in 'interaction_text' ganz regulär mit Text antworten, der zur Interaktion bzw. Anfrage des Nutzers passt.
		  
		"""


		const val SNIPPET_SETTINGS: String =
			"""Sehr gerne, ich kann Ihnen dabei helfen, die Einstellungen anzupassen. 

		Es besteht die Möglichkeit die Sprechgeschwindigkeit der Sprachausgabe anzupassen. Auch ist es möglich die Stimme des Assitentenagenten zu ändern oder aber die Einstellungen zu verlassen. 

		Welche dieser Optionen möchten Sie wählen?"""

		const val SNIPPET_TTS_SPEED: String =
			"""Mit dieser Option können Sie die Sprechgeschwindigkeit der Sprachausgabe anzupassen.
        Der standartmäßige Wert der Geschwindigkeit liegt bei 1,0.
        Möchten Sie die Geschwindigkeit erhöhen, verringern oder auf einen bestimmten Wert setzen?
        """

		const val SNIPPET_VOICE: String =
			"""Mit dieser Einstellung können Sie die Stimme des Assistentenagenten zwischen männlich und weiblich variieren. Möchten Sie die männliche oder die weibliche Assistentenstimme nutzen?
        """
	}


	fun buildOcrPrompt(input: String): String {
		return "Das ist der zuletzt erkannte Text mit den zusätzlichen Koordinaten: " +
			input +
			" \nBitte gib nur diesen in einem Format aus, das es für einen Menschen verständlich macht, der die Daten nur hören, nicht lesen kann." +
			" Überlege dir auch anhand des Kontextes, was der Text tatsächlich aussagen möchte, und korrigiere entsprechende Rechtschreibfehler, wenn nötig und möglich. INTERPRETIERE NICHTS! Wenn es keinen Zusammenhang gibt, dann bleibe bei dem Text, der dir gegeben ist!" +
			" Mache anhand der übergebenen x- und y-Koordinaten des Handybildschirms aus, wo sich der Text in der Kameraperspektive befindet. " +
			" Formuliere den Text so, als würdest du einer Person erklären, wo diese den erkannten Text sieht." +
			" Ein Beispiel wäre: Der Text ... befindet sich links oben von dir aus. Sprich also bitte nicht von einem Bildschirm, sondern sprich diese Person an." +
			" In diesem Fall sollst du anschließend nicht Texterkennung wiederholen bzw. sagen!" +
			" Du solltest niemals in JSON oder einem ähnlichem anderem Format antworten. Antworte so, dass eine blinde Person dich verstehen kann."
	}


	fun generate(command: String, structured: Boolean): String
}