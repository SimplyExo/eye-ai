# EyeAI

**Unscharfe Umgebung, gestochen scharfes Audio.
Das ermöglicht EyeAI: Ein intelligentes Begleitsystem für sehbehinderte Menschen, das die Welt in Echtzeit erkennt, versteht und verständlich über Audio vermittelt. Schnell, privat und lokal auf dem Handy.**  

**Diese App bildet ein lokales lokales Assistenzsystem für Android, das Computer Vision, Objekterkennung, Tracking, Tiefenschätzung, semantische Segmentierung, Spatial Audio, OCR und Sprachsteuerung miteinander kombiniert.**  


## Funktionen

### Echtzeit-Objekterkennung
EyeAI analysiert eingehende Kamerabilder mit einem lokal ausgeführten YOLO-Modell (Yolo26).
Erkannte Objekte werden direkt auf dem Gerät verarbeitet, ohne dass dafür ein externer Cloud-Inferenzdienst benötigt wird.
**Kosten & Latenzen** sind dadurch minimal gehalten, **Privatssphäre & Sicherheit** garantiert.
Inferenzzeiten sind minimal und durch **NPU (Neural Processing Unit) Support & Rust nightly** auf maximale Effizienz getrimmt. 

### Objekt-Tracking mit ByteTrack
EyeAI verwendet ByteTrack, um erkannte Objekte über aufeinanderfolgende Frames hinweg zu verfolgen.
Dadurch werden einzelne Detektionen nicht isoliert betrachtet, sondern können über eine persistente Track-ID einem bereits bekannten Objekt zugeordnet werden.

Das Trackingsystem ermöglicht unter anderem:
- Unterscheidung zwischen neuen und bereits bekannten Objekten
- stabilere Objektausgaben
- Filterung kurzzeitiger oder unsicherer Detektionen
- zeitlich konsistente Objektinformationen
- robustere Verarbeitung bei variabler Inferenzfrequenz
Eine zusätzliche Validierungslogik unterscheidet dabei zwischen noch nicht bestätigten und bestätigten Tracks.

### Tiefenschätzung und Distanzinformationen
EyeAI integriert eine lokale Tiefenschätzung über MiDaS, um zusätzlich zur zweidimensionalen Position eines Objekts auch Informationen über dessen relative Entfernung zu gewinnen.
Das Modell erweitert klassische Objekterkennung damit um räumliche Informationen und bildet eine wichtige Grundlage für situationsabhängiges Audiofeedback.
Auch MiDaS läuft auf unterstützten Handys vollständig auf der **NPU**, wodurch extrem geringe Inferenzzeiten und geringster Energieverbrauch bei bestem Output garantiert bleibt.

### Semantische Segmentierung
Neben der Objekterkennung verfügt EyeAI über semantische Sementierung.
Während die Objekterkennung einzelne relevante Objekte lokalisiert, kann die Segmentierung größere Bildbereiche beziehungsweise semantische Bestandteile der Umgebung klassifizieren.
Dadurch kann EyeAI zusätzliche Informationen über die Struktur einer Szene erfassen, die mit Bounding Boxes von Yolo26 alleine nicht zuverlässig dargestellt werden können.
Beispielsweise ist es so möglich den Boden aus dem SpatialAudio rauszufiltern und stattdessen relevante Gegenstände konsequenter zu betonen.

### Spatial Audio
Erkannte Objekte und räumliche Informationen können über Spatial Audio vermittelt werden.
Dadurch lässt sich nicht nur mitteilen, dass sich ein Objekt in der Umgebung befindet, sondern auch dessen räumliche Position akustisch darstellen.
Die visuelle Umgebung wird dadurch in eine räumlich interpretierbare Audioausgabe übersetzt.
Die Lautstärke eines Audiooutputs ist dabei explizit von der Distanz zum Objekt abhängig.

### Lokale Sprachassistenz
EyeAI verwendet eine eigene, lokale Natural-Language-Processing-Integration zur Interpretation unterstützer Sprachbefehle:
Dazu gehören unter anderem Einstellungen und spezifische Fragen zu Objekten.

Die Verarbeitung kombiniert:
- Intent-Erkennung (Was möchte der Nutzer tun?)
- einen lokalen Settings-Parser (Was möchte der Nutzer verändern und wie?)
- Rückfragen bei Mehrdeutigkeiten
- Bestätigungen vor Änderungen
- State Machine zur Steuerung des Dialogzustands

Frühere Versionen von EyeAI verwendeten zusätzlich als Fallbackoption die Gemini-API.
Diese Abhängigkeit wurde aus der aktuellen Architektur vollständig entfernt!

Die Sprachsteuerung basiert stattdessen auf drei kleinen, selbstentwickelten, lokalen Modellen und deterministischer Logik.

### EyeAIVisionPro 
Neben der Smartphonekamera ist die gesamte EyeAI-Architektur auch für externe Kameras, wie unsere eigens für EyeAIApp angefertigte EyeAIVisionPro, ausgelegt.
Videoframes können über einen externen WebRTC-Stream in die bestehende Analyseprozesse integriert werden.
Kombiniert mit dem Headless Mode liefert dies maximalen Komfort.
Dadurch kann die Kamera perspektivisch unabhängig vom Smartphone positioniert werden, während die eigentliche KI-Verarbeitung weiterhin auf dem mobilen Gerät stattfindet.

Außerdem ermöglicht die EyeAIVisionPro auch Sprachbedienung per Knopfdruck, ohne dabei das Handy überhaupt zu bedienen. 

### Bleeding-edge On-Device-Inferenz
Mehrere KI-Modelle teilen sich auf einem Smartphone begrenzte Ressourcen.
Um alles, was möglich ist, aus einem Smartphone herauszuholen, haben wir es uns zur Aufgabe gemacht auch die allerneusten Architekturoptionen zu nutzen.
Wie verwenden auf Qualcomm-Chips den dedizierten NPU-Chip (Neural Processing Unit Chip), welcher es ermöglicht Rechenoperationen explizit für KI-Modelle deutlich schneller und effizienter auszuführen.
Damit nutzen wir Bleending-edge Technologie zu unserem Vorteil und können gerade deshalb 3 schwere KI-Modelle auf einem Smartphone mit stabilen, hohen FPS-Zahlen ausführen.
Auf Smartphones ohne NPU besteht weiterhin die Option GPU & CPU zugleich zu nutzen. 

### Adaptive Objekterkennung
Die Objekterkennung muss nicht dauerhaft mit maximaler Frequenz ausgeführt werden, ab und zu gibt es auch statische Situationen.
EyeAI passt die Inferenzrate abhängig von der Aktivität der Szene an.
Dadurch kann die benötigte Rechenleistung reduziert werden, wenn sich die visuelle Umgebung nur wenig verändert, während bei relevanten Veränderungen wieder eine höhere Analysefrequenz verwendet wird.

Diese intelligente Steuerung ermöglicht:
- **Reduzierten Energieverbrauch**
- **Weniger thermische Belastung**
- **Weniger unnötige KI-Inferenzen**

Gleichzeitig schafft diese Implementierung Ressourcen für weitere lokale Modelle. 
EyeAI schafft so den Übergang vom Prototypen zur alltagstauglichen Assistenzoption.

### Headless Mode
EyeAI unterstützt einen Headless-Betrieb, bei welchem alle Funktionen auch bei ausgeschaltetem Display weiterlaufen können.
Die App muss damit nicht aktiv vom Nutzer bedient werden, sondern kann vollständig extern gesteuert werden.
Dies reduziert den Energieverbrauch und erhöht den Komfort, insbesondere in Kombination mit der selbstentwickelten EyeAIVisionPro.

### Texterkennung 
EyeAI unterstützt optische Texterkennung über Google OCR Services. Dadurch ist es möglich Texte per Sprachbefehl sofort per Audio ausgeben zu lassen.

### Speech Regonition
EyeAI nutzt Vosk zur schnellen Erkennung und lokalen Verarbeitung von gesprochener, natürlicher Sprache.

Dokumentation von eye-ai-core-rs und dessen API: <https://simplyexo.github.io/eye-ai-docs>

### Performance of EyeAIApp

| **Smartphone**       | **Release date** | **NPU enabled?** | **MiDaS only / inference** | **MiDaS with YOLO / inference**                 |
| -------------------- | ---------------- | ---------------- | -------------------------- | ----------------------------------------------- |
| Samsung Galaxy S25   | 2025             | ✅                | 4.6ms (217 FPS) / 2.25ms   | 12ms (85 FPS), 17ms (58.8 FPS) / 8.5ms, 13ms    |
| Samsung Galaxy S25   | 2025             | ❌                | 16ms (62 FPS) / 12.3ms     | 25ms (40 FPS), 22.4ms (44.6 FPS) / 21.3ms, 13ms |
| Samsung Galaxy S21   | 2021             | ✅ (quantized)    | 10ms (100 FPS) / 2.3ms     | 10ms (100 FPS), 80ms (12.5 FPS) / 3ms, 68ms     |
| Samsung Galaxy S21   | 2021             | ❌                | 48ms (21 FPS) / 34.4ms     | 94ms (10.5 FPS), 90ms (11 FPS) / 85ms, 77ms     |
| Fairphone 4 (no NPU) | 2021             | ❌                | 100ms (10 FPS) / 90ms      | 200ms (5 FPS), 200ms (5 FPS) / 185ms, 175ms     |


### Für ältere Handys mit NPU ist in den Einstellungen dringend empfohlen die quantized MiDaS Version zu verwenden!

## Bedienungsanleitung

### Barrierefreie, interaktive Sprachsteuerung

Als Nutzer kann man mit natürlicher Sprache, nach einem Buttonclick in der App oder auf der EyeAIVision die Spracheingabe starten.
Sofort kommuniziert man dann mit dem Vosk-Modell für Spracherkennung.
Die Kommunikation mit unseren Natural-Language-Processing-Modellen ist als interaktiver Dialog gestaltet und kann intuitiv erfolgen. 
Eine barrierefreien einen Audioguide zur Nutzung der Sprachmodelle findet man sowohl in der App als auch [hier](EyeAIApp/app/src/main/res/raw/nlp_tutorial.wav).

**Die Verarbeitung aller Sprachbefehle läuft vollkommen lokal und privat.**

## Wie from-source compilen?

1. `nix` installieren (siehe <https://nixos.org/download/>, nix der Packagemanager, nicht NixOS das distro, auch wenn NixOS cool ist).

2.	```bash
	nix develop

	cd eye-ai-core
	cargo build-android

	cd ../EyeAIApp/
	./gradlew assembleProduction
	```
Die APK ist dann in `./app/build/outputs/apk/production/app-production.apk`.

Falls man direkt installieren will kann man auch anstatt von `./gradlew assembleProduction` einfach `./gradlew installProduction` ausführen.

(`nix develop` ist nicht nötig wenn sie `direnv` installiert haben)


## Index aller README's:

|                                                                                |                                                                                                                                    |                                                                                                                      |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [Beschreibung](Beschreibung.md)                                                | [EyeAIServer](./EyeAIServer/README.md)                                                                                             | [tflite-runtime litert-gpu third party](./eye-ai-core-rs/tflite-runtime/third_party/litert-gpu-1.2.0/README.md)      |
| [EyeAIApp](./EyeAIApp/README.md)                                               | [EyeAIServer (Server)](./EyeAIServer/Server/README.md)                                                                             | [EyeAIVisionPro ATtiny85_I2C_slave](./EyeAIVisionPro/ATtiny85_I2C_slave/README.md)                                   |
| [EyeAIApp Assets](./EyeAIApp/app/src/main/assets/README.md)                    | [eye-ai-core-rs](./eye-ai-core-rs/README.md)                                                                                       | [EyeAIVisionPro ATtiny85_I2C_slave (master test code)](./EyeAIVisionPro/ATtiny85_I2C_slave/MasterTestCode/README.md) |
| [EyeAIApp StateMachine](./eye-ai-core-rs/doc/StateMachineReadMe.md)            | [eye-ai-core-rs alto fork](./eye-ai-core-rs/alto/README.md)                                                                        | [EyeAIVisionPro gpio_testing_scripts](./EyeAIVisionPro/linux/gpio_testing_scripts/README.md)                         |
| [EyeAIApp OCR](./eye-ai-core-rs/doc/OCRReadMe.md)                              | [eye-ai-core-rs SpatialAudio](./eye-ai-core-rs/doc/SpatialAudioReadMe.md)                                                          | [EyeAIVisionPro drivers](./EyeAIVisionPro/linux/drivers/README.md)                                                   |
| [EyeAIApp Speech Recognition](./eye-ai-core-rs/doc/SpeechRecognitionReadMe.md) | [eye-ai-core-rs Additional Documentation](<./eye-ai-core-rs/doc/Additional Documentation.md>)                                      | [EyeAIVisionPro data gathering](./EyeAIVisionPro/data_gathering/README.md)                                           |
| [EyeAIApp TTS](./eye-ai-core-rs/doc/TTSEngineReadMe.md)                        | [tflite-runtime litert third party](./eye-ai-core-rs/tflite-runtime/third_party/litert-1.2.0/README.md)                            | [EyeAIVision](./EyeAIVision/README.md)                                                                               |
| [EyeAIIcon](./EyeAIIcon/README.md)                                             | [tflite-runtime qnn-litert-delegate third party](./eye-ai-core-rs/tflite-runtime/third_party/qnn-litert-delegate-2.38.0/README.md) | [EyeAIVision (main)](./EyeAIVision/main/README.md)                                                                   |
