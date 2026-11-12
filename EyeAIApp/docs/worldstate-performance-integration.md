# Performance-Integration in exp-worldstate

Quelle: GitHub-Branch `origin/exp-performance-improvements`, Commit `6fc4b15` (lokaler Remote-Ref wegen des wörtlichen Branch-Präfixes: `origin/origin/exp-performance-improvements`). Zielbasis: exp-worldstate `76ed7ce`.

Übernommen: CameraX-Bitmap-Pool mit referenzgezählter Frame-Freigabe, wiederverwendete Modell-Eingabepuffer und Skalierungs-Bitmaps für Tiefe/YOLO/Segmentierung, primitive Rel2Abs-Statistik ohne Float-Boxing, wiederverwendeter Colormap-Arbeitspuffer sowie abschaltbare Segmentierungsvisualisierung.

Kompatibilitätskorrekturen:

- Kamera-Frames behalten CAMERA_TIMESTAMP und skalierte Intrinsics für Worldstate und Rel2Abs.
- Worldstate-Segmentierung, Kontextcache, Recorder und Capture werden unabhängig von der Overlay-Anzeige weiter bedient.
- Veröffentlichte Tiefen-/Segmentierungs-Ausgaben werden nicht wiederverwendet: Audio und weitere Threads besitzen keine zeitlich begrenzten Leases. Ein fester Ring würde nach wenigen Inferenzdurchläufen noch gelesene Daten überschreiben.
- Die UI erhält eigene Debug-/Colormap-Bitmaps statt eines ohne Freigabesignal überschriebenen Bitmap-Rings.
- Der Overlay-Schalter wird geladen, geklont, gespeichert und in der UI nach Neustart korrekt dargestellt.
- Rel2Abs verwendet weiterhin die vorherige Float-Akkumulation für den Mittelwert, um trotz primitiver Sortierung den eingefrorenen Featurevertrag zu erhalten.
- Rotations-Zielbitmaps werden auch bei Fehlern zurückgegeben; Canvas-Verweise werden nach dem Zeichnen gelöst.

Validierung: `./gradlew :app:testDebugUnitTest :app:assembleDebug --offline` erfolgreich: 131 Tests, keine Fehler. Die bestehenden Unit-Tests prüfen unter anderem Worldstate, Kontext, Szenenbeschreibung, Rel2Abs und Audio. Keine neuen Testdateien für diesen Merge erstellt. Hardware-/Mikrofon-/Kamera-/Audio-Laufzeit und tatsächliche FPS/GC-Verbesserungen müssen auf dem Handy geprüft werden. ADB zeigte bei der Integration kein verbundenes Gerät.

Android Studio: vorhandenes Projekt `EyeAIApp` im ursprünglichen Git-Arbeitsbaum öffnen, Branch `exp-worldstate`, Build Variant `debug`, Run-Konfiguration für Modul `app`. USB-Debugging einschalten, Handy verbinden, RSA-Freigabe bestätigen, Gerät auswählen und Run drücken. Der lokale Debug-Build und die vorhandenen nativen Rust-Bibliotheken/Bindings sind vorbereitet. Ein separater Merge des Modus-Routers ist nicht Teil dieser Performance-Integration.
