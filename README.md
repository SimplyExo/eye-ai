# eye-ai

Bilderkennung einer Kamera, die Objekte in der Umgebung in Audio-Hinweise für den Benutzer umwandelt, damit dieser sich ohne Sicht bewegen kann.

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
