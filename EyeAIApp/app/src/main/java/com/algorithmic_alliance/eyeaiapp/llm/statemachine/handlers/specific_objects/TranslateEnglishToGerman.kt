package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp

class TranslateEnglishToGerman {
	companion object {
		private val englishToGermanMap = mapOf(
			"person" to "Person",
			"bicycle" to "Fahrrad",
			"car" to "Auto",
			"motorcycle" to "Motorrad",
			"airplane" to "Flugzeug",
			"bus" to "Bus",
			"train" to "Zug",
			"truck" to "Lkw",
			"boat" to "Boot",
			"traffic light" to "Ampel",
			"fire hydrant" to "Hydrant",
			"stop sign" to "Stoppschild",
			"parking meter" to "Parkuhr",
			"bench" to "Bank",
			"bird" to "Vogel",
			"cat" to "Katze",
			"dog" to "Hund",
			"horse" to "Pferd",
			"sheep" to "Schaf",
			"cow" to "Kuh",
			"elephant" to "Elefant",
			"bear" to "Bär",
			"zebra" to "Zebra",
			"giraffe" to "Giraffe",
			"backpack" to "Rucksack",
			"umbrella" to "Regenschirm",
			"handbag" to "Handtasche",
			"tie" to "Krawatte",
			"suitcase" to "Koffer",
			"frisbee" to "Frisbee",
			"skis" to "Skier",
			"snowboard" to "Snowboard",
			"sports ball" to "Sportball",
			"kite" to "Drachen",
			"baseball bat" to "Baseballschläger",
			"baseball glove" to "Baseballhandschuh",
			"skateboard" to "Skateboard",
			"surfboard" to "Surfbrett",
			"tennis racket" to "Tennisschläger",
			"bottle" to "Flasche",
			"wine glass" to "Weinglas",
			"cup" to "Tasse",
			"fork" to "Gabel",
			"knife" to "Messer",
			"spoon" to "Löffel",
			"bowl" to "Schüssel",
			"banana" to "Banane",
			"apple" to "Apfel",
			"sandwich" to "Sandwich",
			"orange" to "Orange",
			"broccoli" to "Brokkoli",
			"carrot" to "Karotte",
			"hot dog" to "Hot Dog",
			"pizza" to "Pizza",
			"donut" to "Donut",
			"cake" to "Kuchen",
			"chair" to "Stuhl",
			"couch" to "Couch",
			"potted plant" to "Topfpflanze",
			"bed" to "Bett",
			"dining table" to "Esstisch",
			"toilet" to "Toilette",
			"tv" to "Fernseher",
			"laptop" to "Laptop",
			"mouse" to "Maus",
			"remote" to "Fernbedienung",
			"keyboard" to "Tastatur",
			"cell phone" to "Mobiltelefon",
			"microwave" to "Mikrowelle",
			"oven" to "Backofen",
			"toaster" to "Toaster",
			"sink" to "Spüle",
			"refrigerator" to "Kühlschrank",
			"book" to "Buch",
			"clock" to "Uhr",
			"vase" to "Vase",
			"scissors" to "Schere",
			"teddy bear" to "Teddybär",
			"hair drier" to "Haartrockner",
			"toothbrush" to "Zahnbürste"
		)

		fun translateToGerman(englishLabel: String): String {
			val translation = englishToGermanMap[englishLabel] ?: englishLabel
			Log.d(EyeAIApp.Companion.APP_LOG_TAG, "TranslateToGerman: '$englishLabel' -> '$translation'")
			return translation
		}

		fun isKnownEnglishLabel(englishLabel: String): Boolean =
			englishLabel in englishToGermanMap
	}
}
