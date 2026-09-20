# EyeAI Intent – Retraining- und Evaluationsplan

## 1. Dateninventar
- Alter Bestand: `DATASET.train` (3.627 Beispiele) und `DATASET.val` (1.033 Beispiele).
- Person 1: 81 eindeutige Satz/Gold-Paare.
- Person 2: byte-/inhaltlich dieselben Daten wie Person 1 → nicht doppelt zählen.
- Person 3: 100 Zeilen, 92 eindeutige Satz/Gold-Paare.
- Person 4: 258 Zeilen, 256 eindeutige Satz/Gold-Paare.
- Person 5: Kopie von Person 4 → nicht doppelt zählen.

## 2. Aktuelle Modelle
Bewertet wurden nur die drei tatsächlich verschiedenen Personensätze und pro Datei interne Satzduplikate entfernt.

Mittlerer Macro-F1 über Person 1, 3 und 4:
- M0-Familie: ca. 73,7 %
- M1-Familie: ca. 76,9 %
- M2-Familie: ca. 73,8 %
- M3-Familie: ca. 72,0 %

Bester einzelner Lauf:
- M1_T1: ca. 77,5 % Accuracy / 78,8 % Macro-F1 im Mittel über die drei unterschiedlichen Personensätze.

Damit ist M1_T1 die sinnvollste aktuelle Referenz vor dem Retraining.

## 3. Warum nicht einfach alle neuen Sätze ins Training?
Der alte Trainingsbestand enthält bereits besonders viele Beispiele für `SET_BPS`, `SET_FREQUENCY` und `MEASURE_DISTANCE`.
Die neuen Fehler zeigen daher vor allem Grenzflächen-/Formulierungsprobleme, nicht bloß Datenmangel.
Daher werden nur gezielte, klare Fehler- und Robustheitsbeispiele ergänzt.

## 4. Neue Splits

### Training additions
`intent_new_additions.train`
- 24 gezielte Beispiele.
- Priorisiert wiederkehrende Fehler, neue natürliche Formulierungen und fehlende Cue-Varianten.
- Bewusst keine redundante Auffüllung von Klassen, die in den neuen Daten bereits stabil laufen.

### Validation
`intent_new_validation.val`
- 36 Beispiele aus Person 3.
- Kein normalisiertes exaktes Duplikat zum alten Train/Val oder zu den neuen Trainingsergänzungen.
- Mischung aus normalen und einigen schwierigeren Formulierungen.
- Dient für Modell-/Seed-/Hyperparameterwahl, nicht als finale Leistungszahl.

### Blind Core
`intent_blind_core.inputs.txt` + `intent_blind_core.gold`
- 60 Beispiele aus Person 4.
- Genau 6 pro Intentklasse.
- Keine normalisierten exakten Duplikate zum bisherigen Train/Val oder zu den neuen Train/Validation-Splits.
- Repräsentativer Mix aus kurzen, langen und umgangssprachlichen Formulierungen.
- Primärer Benchmark nach dem Retraining.

### Blind Hard
`intent_blind_hard.inputs.txt` + `intent_blind_hard.gold`
- 20 zusätzliche Challenge-Beispiele aus Person 4.
- Bewusst auf bekannte Schwachstellen konzentriert: BPS/Frequency-Grenzen, lange OCR-Kommandos, OOD-vs-Distanz, schwierige Abbruchformulierungen.
- Nicht als alleinige Hauptmetrik verwenden; getrennt vom Blind Core berichten.

## 5. Wichtige Labelprobleme
Siehe `intent_manual_review.md`.
Besonders wichtig:
- Lautstärke ist nicht Frequenz/Tonhöhe.
- Signalton-Geschwindigkeit ist BPS, nicht TTS-Sprechgeschwindigkeit.
- `Sprachausgabengeschwindigkeit` besitzt in Person 4 widersprüchliche Gold-Labels und wird ausgeschlossen.
- Unterbestimmte Phrasen werden nicht künstlich zu einer Settings-Unterklasse gezwungen.

## 6. Empfohlener Ablauf
1. Aktuellen Stand einfrieren und M0–M3 auf denselben sauberen Baselines dokumentieren.
2. `DATASET.train` + `intent_new_additions.train` zum Retraining verwenden.
3. Modellvarianten ausschließlich mit bestehender Validation + `intent_new_validation.val` vergleichen.
4. Kandidaten einfrieren.
5. Erst dann `intent_blind_core` auswerten.
6. `intent_blind_hard` separat als Robustheitsdiagnose auswerten.
7. Primäre Auswahlmetrik: Macro-F1 auf Blind Core; Accuracy und per-class F1 zusätzlich berichten.
8. Konfusionsmatrix besonders für `SET_BPS`↔`CHANGE_SPEECH_SPEED`↔`SET_FREQUENCY`, `TEXT_RECOGNITION`↔`OBJECT_DETECTION` und `REDIRECT_TO_LLM` ausgeben.
9. Nach Auswahl des Produktionsmodells den restlichen unbenutzten Person-4-Pool als Reserve für einen letzten Audit behalten.
