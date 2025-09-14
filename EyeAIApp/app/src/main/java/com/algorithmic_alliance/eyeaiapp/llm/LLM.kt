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
		  
		  3. Objekterkennung:
		  Verwende dies NUR wenn der Nutzer explizit nach einem spezifischen Objekt fragt:
			- "Wo ist der Stuhl?" / "Wie weit ist die Lampe entfernt?" 
			- "Beschreibe mir den Tisch" / "Wo befindet sich die Person?"
		
		  WICHTIG: Der Nutzer MUSS ein spezifisches Objekt nennen!
		  Übersetze das Objekt ins Englische und trage es in "object_query" ein.
		  Beispiele: "Stuhl" -> "chair", "Lampe" -> "lamp", "Tisch" -> "table", "Person/Mensch" -> "person"
	
		  WENN objekterkennung in JSON true ist, dann muss auch object_query gesetzt werden!

		  Wenn der Nutzer nur allgemein fragt ("Was sehe ich?", "Objekterkennung"), so wird dieses Tool verwendet. Dies wird durch die Eigenschaft 'objekterkennung' im JSON gesteuert.
		  
		  
		  Möchte der Nutzer keine der Tools nutzen, so kannst du auch in der JSON-Antwort in 'interaction_text' ganz regulär mit Text antworten, der zur Interaktion bzw. Anfrage des Nutzers passt.
		  
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

		val knownObjectLabels = setOf(
			"airplane", "ambulance", "barge", "bathroom cabinet", "bathtub", "bed", "bench",
			"bicycle", "bidet", "billboard", "boat", "bookcase", "boy", "building", "bus",
			"cabinetry", "car", "castle", "cat furniture", "chair", "chest of drawers",
			"closet", "coffee table", "couch", "countertop", "cupboard", "desk", "dishwasher",
			"door", "door handle", "drawer", "filing cabinet", "furniture", "gas stove",
			"girl", "golf cart", "gondola", "helicopter", "home appliance", "humidifier",
			"infant bed", "jet ski", "kitchen & dining room table", "lamp", "land vehicle",
			"light bulb", "light switch", "lighthouse", "limousine", "loveseat", "man",
			"microwave oven", "motorcycle", "nightstand", "oven", "person", "porch",
			"power plugs and sockets", "refrigerator", "shelf", "shower", "sink", "skyscraper",
			"soap dispenser", "sofa bed", "stairs", "stool", "stop sign", "street light",
			"studio couch", "submarine", "table", "tank", "taxi", "toilet", "tower",
			"traffic light", "traffic sign", "train", "training bench", "truck", "unicycle",
			"van", "vehicle", "wall clock", "wardrobe", "washing machine", "window",
			"window blind", "woman"
		)
	}

	fun buildObjectDetectionPrompt(label: String, height: Float, width: Float, x: Float, y: Float, distance: Float): String {

		return "Vor dir befindet sich das Objekt '$label'. " +
			"Der sehbeinträchtigte Nutzer möchte von dir wissen, wo in etwa sich das Objekt von dir aus gesehen befindet. " +
			"Dazu hast du folgende die folgenden Daten: " +
			"Das ist die x-Koordinate der Mitte des Objekts $x, das ist die y-Koordinate der Mitte des Objekts $y. " +
			"Das ist die Höhe des Objekts $height. " +
			"Das ist die Breite des Objekts $width. "+
			"Das ist die Distanz zum Objekt aus der Perspektive des Nutzers: $distance " +
			"Antworte nicht mit einem JSON-Objekt, sondern mit Standartsprache!"
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