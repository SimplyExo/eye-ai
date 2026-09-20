# EyeAI Intent - Beispiele, die vor Training NICHT ungeprüft übernommen werden sollten

## Person 1
- `wert ändern` → OPEN_SETTINGS: zu unspezifisch.
- `Ton schneller machen` → SET_BPS: kann Sprechtempo, BPS oder Tonhöhe bedeuten.
- `schnellere Tonausgabe` → SET_BPS: ebenfalls mehrdeutig.
- `langsame Tonausgabe bitte` → SET_BPS: besonders stark mit CHANGE_SPEECH_SPEED überlappend.
- `Der ton tut weh` → SET_FREQUENCY: kann Lautstärke statt Tonhöhe meinen.
- `Tonausgabe ändern` → SET_FREQUENCY: unterbestimmt.
- `was soll das denn sein` → OBJECT_DETECTION: ohne visuellen Kontext auch rhetorisch/allgemein interpretierbar.
- Unvollständige Redirect-Fragmente (`Bitter erkläre mir ein`, `erzähle mir über`, `erläutere mir das prinzip von`, `Wie funktioniert`) nur als explizit markierte ASR-/Fragmentdaten verwenden.

## Person 3
- `test` → ABORT: semantisch nicht belastbar.
- `ändere deine sprache`, `andere sprache` → CHANGE_SPEAKER: Sprache und Sprecherstimme sind nicht dasselbe.
- `was hast du gesagt` → CHANGE_SPEECH_SPEED: eher Wiederholung/Nachfrage als Geschwindigkeitsänderung.
- `wiederherstellen` → ABORT: semantisch eher Restore/Undo.
- `puls erhöhen`, `puls verlangsamen` → SET_BPS: nur verwenden, wenn „Puls“ im Produkt ausdrücklich als BPS-Synonym definiert ist.

## Person 4 / 5
- `Sprachausgabengeschwindigkeit` besitzt im Quelldatensatz zwei widersprüchliche Gold-Labels (CHANGE_SPEECH_SPEED und TEXT_RECOGNITION) und wird deshalb ausgeschlossen.
- Lautstärke-Beispiele unter SET_FREQUENCY (`Ich höre kaum etwas`, mehrere Sätze mit „Lautstärke“) nicht übernehmen: Lautstärke ≠ Frequenz/Tonhöhe.
- `signal ton geschwindigkeit ändern` unter CHANGE_SPEECH_SPEED nicht übernehmen: beschreibt eher SET_BPS.
- Unterbestimmte Phrasen wie `Kannst du die Signaltöne anpassen` oder `Lässt sich das Piepen irgendwie verändern` nicht als eindeutiges SET_FREQUENCY-Gold verwenden.
