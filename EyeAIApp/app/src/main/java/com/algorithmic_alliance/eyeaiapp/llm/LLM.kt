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
		  - Der Nutzer möchte die Frequenz (bzw. die Tonhöhe!) der Distanztöne anpassen
		  - Der Nutzer möchte die Schläge pro Sekunden/Beats per second der Distanztöne anpassen
		  - Der Nutzer möchte die Einstellungen öffnen oder aufrufen
		  
		  So wird dieses Tool verwendet. Dies wird durch die Eigenschaft 'einstellungen' im JSON gesteuert.
          Innerhalb der Einstellungen wird die genaue Absicht des Nutzers über die Eigenschaft 'setting_intent' im JSON klassifiziert.
		  
		  3. Objekterkennung:
		  Verwende dies NUR wenn der Nutzer explizit nach einem spezifischen Objekt fragt:
			- "Wo ist der Stuhl?" / "Wie weit ist die Lampe entfernt?" 
			- "Beschreibe mir den Tisch" / "Wo befindet sich die Person?"
		  
		  WICHTIG: Der Nutzer MUSS ein spezifisches Objekt nennen!
		  Schreibe das deutsche Objektname EXAKT in "object_query" ein.
			
		  ERLAUBTE DEUTSCHE OBJEKTNAMEN:
Person, Fahrrad, Auto, Motorrad, Flugzeug, Bus, Zug, Lkw, Boot, Ampel, Hydrant, Stoppschild, Parkuhr, Bank, Vogel, Katze, Hund, Pferd, Schaf, Kuh, Elefant, Bär, Zebra, Giraffe, Rucksack, Regenschirm, Handtasche, Krawatte, Koffer, Frisbee, Skier, Snowboard, Sportball, Drachen, Baseballschläger, Baseballhandschuh, Skateboard, Surfbrett, Tennisschläger, Flasche, Weinglas, Tasse, Gabel, Messer, Löffel, Schüssel, Banane, Apfel, Sandwich, Orange, Brokkoli, Karotte, Hot Dog, Pizza, Donut, Kuchen, Stuhl, Couch, Topfpflanze, Bett, Esstisch, Toilette, Fernseher, Laptop, Maus, Fernbedienung, Tastatur, Mobiltelefon, Mikrowelle, Backofen, Toaster, Spüle, Kühlschrank, Buch, Uhr, Vase, Schere, Teddybär, Haartrockner, Zahnbürste

Wähle das passendste Objekt aus der Liste basierend auf der Benutzeranfrage:

Beispiele:
- "Wo ist die Person?" → {"requested_functions": {"objekterkennung": true}, "object_query": "Person"}
- "Zeig mir das Auto" → {"requested_functions": {"objekterkennung": true}, "object_query": "Auto"}  
- "Wo befindet sich der Computer?" → {"requested_functions": {"objekterkennung": true}, "object_query": "Laptop"}
- "Ich suche den Stuhl" → {"requested_functions": {"objekterkennung": true}, "object_query": "Stuhl"}

Verwende NUR Objektnamen aus der obigen Liste! Wenn der Benutzer nach etwas fragt, das nicht in der Liste steht, wähle das ähnlichste Objekt oder antworte über interaction_text.

		  Wenn der Nutzer nur allgemein fragt ("Was sehe ich?", "Objekterkennung"), so wird dieses Tool verwendet. Dies wird durch die Eigenschaft 'objekterkennung' im JSON gesteuert.
		  
		  
		  Möchte der Nutzer keine der Tools nutzen, so kannst du auch in der JSON-Antwort in 'interaction_text' ganz regulär mit Text antworten, der zur Interaktion bzw. Anfrage des Nutzers passt.
		  
		"""


		const val SNIPPET_SETTINGS: String =
			"""Sehr gerne, ich kann Ihnen dabei helfen, die Einstellungen anzupassen. 

		Es besteht die Möglichkeit die Sprechgeschwindigkeit der Sprachausgabe anzupassen, die Stimme des Assitentenagenten zu ändern, die Tonhöhe der Distanztöne zu ändern,
		die Schläge pro Sekunde für die Distanzhinweistöne zu ändern, oder aber die Einstellungen zu verlassen. 

		Welche dieser Optionen möchten Sie wählen?"""

		const val SNIPPET_TTS_SPEED: String =
			"""Mit dieser Option können Sie die Sprechgeschwindigkeit der Sprachausgabe anpassen.
        Der standartmäßige Wert der Geschwindigkeit liegt bei 1,0.
        Möchten Sie die Geschwindigkeit erhöhen, verringern oder auf einen bestimmten Wert setzen?
        """

		const val SNIPPET_VOICE: String =
			"""Mit dieser Einstellung können Sie die Stimme des Assistentenagenten zwischen männlich und weiblich variieren. Möchten Sie die männliche oder die weibliche Assistentenstimme nutzen?
        """



		const val SNIPPET_BPS: String = "Mit dieser Einstellung können Sie die Distanzhinweisschläge pro Sekunde anpassen. Möchten Sie die Frequenz dieser erhöhen oder verringern?"

		const val SNIPPET_FREQUENCY: String = "Mit dieser Einstellung können sie Tonhöhe der Distanzhinweisetöne anpassen. Möchten Sie diese erhöhen oder verringern?"
		val knownObjectLabels = setOf(
			"person",
			"bicycle",
			"car",
			"motorcycle",
			"airplane",
			"bus",
			"train",
			"truck",
			"boat",
			"traffic light",
			"fire hydrant",
			"stop sign",
			"parking meter",
			"bench",
			"bird",
			"cat",
			"dog",
			"horse",
			"sheep",
			"cow",
			"elephant",
			"bear",
			"zebra",
			"giraffe",
			"backpack",
			"umbrella",
			"handbag",
			"tie",
			"suitcase",
			"frisbee",
			"skis",
			"snowboard",
			"sports ball",
			"kite",
			"baseball bat",
			"baseball glove",
			"skateboard",
			"surfboard",
			"tennis racket",
			"bottle",
			"wine glass",
			"cup",
			"fork",
			"knife",
			"spoon",
			"bowl",
			"banana",
			"apple",
			"sandwich",
			"orange",
			"broccoli",
			"carrot",
			"hot dog",
			"pizza",
			"donut",
			"cake",
			"chair",
			"couch",
			"potted plant",
			"bed",
			"dining table",
			"toilet",
			"tv",
			"laptop",
			"mouse",
			"remote",
			"keyboard",
			"cell phone",
			"microwave",
			"oven",
			"toaster",
			"sink",
			"refrigerator",
			"book",
			"clock",
			"vase",
			"scissors",
			"teddy bear",
			"hair drier",
			"toothbrush"
		)
	}

	fun buildObjectDetectionPrompt(label: String, height: Float, width: Float, x: Float, y: Float, distance: Float): String {
		return "Vor dir befindet sich das Objekt '$label'. " +
			"Der sehbehinderte Nutzer möchte von dir wissen, wo in etwa sich das Objekt von dir aus gesehen befindet. " +
			"Dazu hast du folgende Daten: " +
			"Das ist die x-Koordinate der Mitte des Objekts $x, das ist die y-Koordinate der Mitte des Objekts $y. " +
			"Das ist die Höhe des Objekts $height. " +
			"Das ist die Breite des Objekts $width. " +
			"Das ist die Distanz zum Objekt aus der Perspektive des Nutzers: $distance Meter. " +
			"Erstelle eine hilfreiche, natürliche Antwort auf Deutsch, die dem Benutzer nur die Position und Details des angefragten Objekts '$label' erklärt. " +
			"Erwähne keine anderen Objekte. Konzentriere dich ausschließlich auf das angefragte Objekt. " +
			"Die Antwort sollte freundlich, präzise und für sehbehinderte Menschen hilfreich sein. " +
			"Verwende natürliche Sprache und vermeide technische Koordinaten-Details. " +
			"Antworte nicht mit einem JSON-Objekt, sondern mit Standardsprache!"
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

	fun buildSettingsMenuPrompt(input: String) = """Der Nutzer ist im Einstellungsmenü und sagt: $input.
    
		Klassifiziere die Absicht des Nutzers in eine der folgenden Kategorien und gib sie im Feld 'setting_intent' zurück:
		
		- 'tts_speed': Wenn der Nutzer die Sprechgeschwindigkeit ändern will (z.B. "schneller sprechen").
		- 'voice': Wenn der Nutzer die Stimme des Assistenten ändern will (z.B. "Stimme ändern", "andere Stimme", "Assistentenagenten anpassen").
		- 'frequency': Wenn der Nutzer die Audio-Frequenz ändern will (z.B. "Frequenz anpassen", "Tonhöhe ändern").
		- 'bps': Wenn der Nutzer die BPS (Beats per Second) ändern will (z.B. "BPS ändern", "Schläge pro Sekunde").
		- 'leave': Wenn der Nutzer die Einstellungen verlassen will.
		- 'none': Wenn keine der obigen Absichten klar erkennbar ist.
		
		Antworte NUR mit dem JSON-Objekt.
		
		Beispiel für die Eingabe "ich will eine andere Stimme": {"setting_intent": "voice"}
		Beispiel für die Eingabe "Frequenz ändern": {"setting_intent": "frequency"}
		Beispiel für die Eingabe "verlassen": {"setting_intent": "leave"}
		"""

	fun generate(command: String, structured: Boolean): String
}