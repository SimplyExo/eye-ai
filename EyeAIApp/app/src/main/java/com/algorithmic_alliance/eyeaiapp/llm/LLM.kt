package com.algorithmic_alliance.eyeaiapp.llm

import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.ocr.GoogleOCR
import com.algorithmic_alliance.eyeaiapp.ocr.OCRManager

interface LLM {
	companion object {
		const val SYSTEM_PROMPT: String =
			"""Du bist ein Sprachassistent, welcher gesprochene Befehle bekommt und anhand dieser bestimmte Tools verwendet, welche zum gesprochenen Befehl passen.

		Die gesprochenen Befehle können möglicherweise fehlerhaft erkannt werden.
		Verwende den Kontext, um mögliche Fehler zu ignorieren und korrekt zu antworten.
		Frage dabei nicht nach einer Klarifikation durch den User, sondern gehe vom Wahrscheinlichsten aus, was der User meinen könnte.
		Rufe den User nicht auf, sich zu wiederholen!

		Du hast folgende Tools:

		1. Texterkennung:
		Wenn der Nutzer einen Text aus dem Kamerabild vorgelesen haben will, wird dieses Tool verwendet.
	
		2. Einstellungen:
		Wenn der Nutzer die Einstellungen der Text-zu-Sprache-Instanz anpassen möchte, wie beispielsweise die Lautstärke oder aber die Sprechgeschwindigkeit, so wird dieses Tool verwendet.
	

		Um ein Tool zu verwenden, musst du den Namen des Tools am Ende deiner Antwort nennen.
		Nur der Name des Tools, nichts Weiteres!
		Solltest du das Tool zuletzt genutzt haben und die Werte schon erhalten haben, so wiederhole in deiner Antwort nicht mehr den Namen des Tools!
		"""

		const val SETTINGS_PROMPT: String = """
			Der Nutzer möchte die Einstellungen anpassen. Er hat folgende Möglichkeiten: 
			1) Sprachgeschwindigkeit der Sprachausgabe anpassen
			2) Assistentenstimme anpassen	(dies ermöglicht dem Nutzer nur zu wählen, welche Stimme der TTS Instanz er nutzen möchte, nur dies sollten deine künftigen Erklärungen auch beeinhalten!)
			3) Er kann die Einstellungen verlassen
			
			Erkläre dich einverstanden damit und diktiere ihm diese bitte in dieser Reihenfolge, reihe diese bitte mit erstens, zweitens, drittens und viertens aneinander.
			Frage ihn dann, welche Option er wählen möchte.
			Merke dir, das der Nutzer soeben die Einstellungen aufgerufen hat. Nennt der Nutzer im nächsten Befehl eine dieser Optionen, so schreibe nur die Zahl der Option als deine nächste Antwort.
			Bei "Sprachgeschwindigkeit der Sprachausgabe anpassen" wäre dies also 1.
			In diesem Fall sollst du anschließend nicht Einstellungen wiederholen bzw. sagen!
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
			" In diesem Fall sollst du anschließend nicht Texterkennung wiederholen bzw. sagen!"
	}


	fun generate(command: String, structured: Boolean): String
}