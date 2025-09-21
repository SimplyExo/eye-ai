# eye-ai

Bilderkennung einer Kamera, die Objekte in der Umgebung in Audio-Hinweise für den Benutzer umwandelt, damit dieser sich ohne Sicht bewegen kann.

Dokumentation in EyeAI Docs: <https://simplyexo.github.io/eye-ai-docs>

### Performance of EyeAIApp

| **Smartphone**       | **Release date** | **NPU enabled?** | **MiDaS only / inference** | **MiDaS with YOLO / inference**                 |
| -------------------- | ---------------- | ---------------- | -------------------------- | ----------------------------------------------- |
| Samsung Galaxy S25   | 2025             | ✅               | 4.6ms (217 FPS) / 2.25ms   | 12ms (85 FPS), 17ms (58.8 FPS) / 8.5ms, 13ms    |
| Samsung Galaxy S25   | 2025             | ❌               | 16ms (62 FPS) / 12.3ms     | 25ms (40 FPS), 22.4ms (44.6 FPS) / 21.3ms, 13ms |
| Samsung Galaxy S21   | 2021             | ✅ (quantized)   | 10ms (100 FPS) / 2.3ms     | 10ms (100 FPS), 80ms (12.5 FPS) / 3ms, 68ms     |
| Samsung Galaxy S21   | 2021             | ❌               | 48ms (21 FPS) / 34.4ms     | 94ms (10.5 FPS), 90ms (11 FPS) / 85ms, 77ms     |
| Fairphone 4 (no NPU) | 2021             | ❌               | 100ms (10 FPS) / 90ms      | 200ms (5 FPS), 200ms (5 FPS) / 185ms, 175ms     |

# Bedienungsanleitung

## Sprachbefehle nutzen

Als Nutzer kann man mit natürlicher Sprache, nach einem Buttonclick in der App oder auf der EyeAIVision die Spracheingabe starten.
Sofort kommuniziert man dann mit dem Vosk-Modell für Spracherkennung. Gibt man nun einen beliebigen Input, so erfolgt eine kurze Klassifizierung des Inputs.

## Klassifizierungen

Es gibt verschiedene Möglichkeiten, wie unsere Modelle Input klassifizieren können.

### 1. Einstellungen
Gelangt man in die Einstellungen, so ist das Einstellungsmenü als interaktiver Dialog gestaltet, der alle Einstellungsoptionen diktiert und anschließend für jede Einstellung einen Anpassungsdialog bietet.

### 3. Texterkennung
Die Texterkennung erfolgt sofort auf Sprachbefehl sofern Text erkannt werden konnte.

### 4. Objektinformationen
Objektinformationen erfolgen ebenfalls, sofern das jeweilige angefragte Objekt erkannt wurde, sofort auf Befehl. Notwendig bei der Anfrage ist es dabei jedoch das gewünschte Objekt zu nennen.

### 5. Generelle Fragen
Wissensfragen, Fragen zum Alltag o.ä. werden hier beantwortet. Die Antwort findet sofort statt.
