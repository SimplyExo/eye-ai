package com.algorithmic_alliance.eyeaiapp.llm

import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.ocr.GoogleOCR
import com.algorithmic_alliance.eyeaiapp.ocr.OCRManager

interface LLM {
	companion object {
		const val SYSTEM_PROMPT: String =
			"""Du bist ein Sprachassistent, welcher gesprochene Befehle bekommt, und anhand dieser bestimmte Tools verwendet, welche zum gesprochenen Befehl passen.

Die gesprochenen Befehle können möglicherweise fehlerhaft erkannt werden.
Verwende den Kontext, um mögliche Fehler zu ignorieren, und korrekt zu antworten.
Frage dabei nicht nach einer Klarifikation durch den Users, sondern gehe von dem Wahrscheinlichstem aus, was der User meinen könnte.
Rufe den User nicht auf, sich zu wiederholen!

Du hast folgende Tools:

1. Texterkennung:
Wenn der Nutzer einen Text aus dem Kamerabild vorgelesen haben will, wird dieses Tool verwendet.


Um ein Tool zu verwenden, musst du den Namen des Tools am Ende deiner Antwort nennen.
Nur der Name des Tools, nichts weiteres!
Solltest du das Tool zuletzt genutzt haben und die Werte schon erhalten haben, so wiederhole in deiner Antwort nicht mehr den Namen des Tools!"""


	}

	fun buildOcrPrompt(input: String): String {
		return "Das ist der zuletzt erkannte Text mit den zusätzlichen Koordinaten: " +
			input +
			" \nBitte gib nur diesen in einem Format aus, dass es für einen menschen verständlich macht, der die Daten nur hören, nicht lesen kann." +
			" Überlege dir auch anhand des Kontextes, was der Text tatsächlich aussagen möchte und korrigiere entsprechende Rechtschreibfehler wenn nötig und möglich. INTERPRETIERE NICHTS! Wenn es keinen Zusammenhang gibt, dann bleibe bei dem Text der dir gegeben ist!" +
			" Mache anhand der übergebenen x und y Koordinaten des Handybildschirms aus, wo sich der Text in der Kameraperspektive befindet. " +
			" Formuliere den Text so, als würdest du einer Person erklären, wo diese den erkannten Text sieht." +
			" Ein Beispiel wäre: Der Text ... befindet sich links oben von dir aus. Sprich also bitte nicht von einem Bildschirm, sondern sprich diese Person an." +
			" Nur in diesem Fall sollst du anschließend nicht Texterkennung wiederholen bzw. sagen!"
	}

	suspend fun generate(prompt: String): String
}