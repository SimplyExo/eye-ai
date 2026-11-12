# Sprachgesteuerter Moduswechsel

Basis: `exp-worldstate` (76ed7ce), weil dieser Stand die drei Ausgabemodi und die lokale Szenenbeschreibung enthält.

Der separate vierklassige Router läuft im Ruhezustand und im gesprochenen Einstellungsmenü vor dem bestehenden zehnklassigen Intentmodell. Nur wenn seine stärkste Klasse ein Modus ist und mindestens 0.3971867859363556 erreicht, wird die gespeicherte Ausgabemodus-Einstellung geändert. NO_MODE, zu geringe Sicherheit oder ein Routerfehler führen zum bisherigen Intentpfad. Der Schwellenwert von 0,6 des Intentmodells wird nicht auf Routerausgaben angewendet. Bestätigungsdialoge und Eingaben innerhalb einer konkreten Einstellungsänderung behalten ihre bisherige Zuständigkeit.

Die Auswahl wirkt über `Settings.save` und `EyeAIApp.updateSettings` auf die vorhandene Runtime: Szenenbeschreibung einschalten bzw. beenden und Normal-/Objectwatch-Audio konfigurieren. Der alte Einstellungszustand bleibt bis zum Runtime-Update erhalten, damit die Runtime den Wechsel erkennt. Die drei manuellen Ausgabemodi bleiben im Audio-Einstellungsbereich verfügbar. Unter Spracherkennung gibt es den standardmäßig eingeschalteten Schalter „Modus per Sprache wechseln“. Seine Auswahl wird gespeichert und beim Neustart geladen. Ein akzeptierter Sprachbefehl wird gesprochen bestätigt; im Einstellungsmenü bleibt dieses geöffnet.

## Modellvertrag

- Quelle: Experiment `mode_switch_v3_architecture_finetune/separate_mode_router`, Seed 20260812 (nach Dev-Loss ausgewählt).
- Labels: MODE_NORMAL, MODE_OBJECT_WATCH, MODE_SCENE_DESCRIPTION, NO_MODE.
- Unveränderter App-T2-BPE-Tokenizer, 24 int32-Token, Ausgabe vier float32-Wahrscheinlichkeiten.
- `assets/nlp-v2/mode-router/config.json` enthält Schwellenwert, Labels, Seed und Modell-SHA256. Beim Laden werden Hash, Labelreihenfolge, Tokenizer und Tensorvertrag geprüft.
- Intentmodell und dessen zehn Klassen sind unverändert. Router und Intentmodell werden gemeinsam geladen und geschlossen.

Die bekannten Offline-Blindwerte sind 89,74% auf Modus-Clean und 85,64% auf synthetischer Modus-ASR. Die Kaskade erreicht 78,33% auf menschlichem Intent-Clean und 73,33% auf menschlichem Intent-ASR; falsche Moduswechsel bleiben möglich. Der im Dev-Split begrenzte Falschpositivanteil ist keine Garantie für andere Eingaben.

## Prüfung

`./gradlew :app:testDebugUnitTest :app:assembleDebug :app:assembleProduction --offline`

Lokal bestanden 133 Unit-Tests, darunter fünf nicht versionierte Router-Tests. Die neuen Testdateien sind auf Nutzerwunsch nicht Teil dieses Commits.

Die lokal ausgeführten Router-Unit-Tests prüfen Klassenreihenfolge, NO_MODE-Rückfall, Ablehnung unsicherer Ausgaben, den separaten Schwellwert und ungültige Modellwerte. Eine Offline-Inferenz mit dem gebündelten Modell erkennt „Szenenbeschreibung“, „Normaler Modus“ und „Objectwatch“ als die jeweiligen Modi. „Lies mir den Text vor“ und „Öffne die Einstellungen“ ergeben NO_MODE. Native Bibliothek und generierte Rust-Bindings wurden für den lokalen Build aus dem identischen Basisbranch übernommen; sie sind wie bisher generierte Build-Voraussetzungen. Ein echter Mikrofon-/ASR-/Audio-Gerätetest ist separat erforderlich.
