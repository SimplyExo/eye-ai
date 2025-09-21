# eye-ai

Bilderkennung einer Kamera, die Objekte in der Umgebung in Audio-Hinweise für den Benutzer umwandelt, damit dieser sich ohne Sicht bewegen kann.

### Performance of EyeAIApp

| **Smartphone**       | **Release date** | **NPU enabled?** | **MiDaS only / inference** | **MiDaS with YOLO / inference**                 |
| -------------------- | ---------------- | ---------------- | -------------------------- | ----------------------------------------------- |
| Samsung Galaxy S25   | 2025             | ✅               | 4.6ms (217 FPS) / 2.25ms   | 12ms (85 FPS), 17ms (58.8 FPS) / 8.5ms, 13ms    |
| Samsung Galaxy S25   | 2025             | ❌               | 16ms (62 FPS) / 12.3ms     | 25ms (40 FPS), 22.4ms (44.6 FPS) / 21.3ms, 13ms |
| Samsung S21          | 2021             | ✅ (quantized)   | 10ms (100 FPS) / 2.3ms     | 10ms (100 FPS), 80ms (12.5 FPS) / 3ms, 68ms     |
| Samsung S21          | 2021             | ❌               | 48ms (21 FPS) / 34.4ms     | 94ms (10.5 FPS), 90ms (11 FPS) / 85ms, 77ms     |
| Fairphone 4 (no NPU) | 2021             | ❌               | 100ms (10 FPS) / 90ms      | 200ms (5 FPS), 200ms (5 FPS) / 185ms, 175ms     |

### Projekt-Plan:

#### 1. Schritt:

Erkennung von Objekten im Raum ohne Klassifizierung jedoch mit Messung der Entfernung. Dann Umwandlung in Ton mit Richtung.
Wenn das Objekt nicht erkannt wird, wird stattdessen ein solider Block dahin gestellt, um zu verhindern, dass der Nutzer in etwas läuft.
Not-Aus: Wenn das Programm nicht mitkommt, wird der Nutzer gewarnt, er solle sich erstmal nicht weiter bewegen, bis das Programm aufgehohlt hat.

#### 2. Schritt:

Klassifizierung von Objekten, möglicherweise Ausgabe per Sprache.
Möglichkeit der Ausgabe der Objekte im aktuellen Sichtfeld des Nutzers auf dessen Eingabe hin.

#### 3. Schritt

Gesichtserkennung von bekannten Personen?
Nachtsicht?
