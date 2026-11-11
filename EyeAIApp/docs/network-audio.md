# Audioausgabe über EyeAIVision

Bei laufendem Betrieb mit der Eingabequelle **EyeAIVision** sendet die App ihren
gemeinsamen Audiomix per TCP an **192.168.4.1:4041**. Dazu gehören TTS, räumliche
Objekt- und Tiefentöne, die Vosk-Signaltöne und die Tutorial-Aufnahmen. Bei anderen
Eingabequellen erfolgt die Ausgabe lokal. Beim Beenden des Betriebs wird die
Verbindung geschlossen.

## Drahtformat

Der Empfänger erhält fortlaufend **48.000 Stereo-Frames pro Sekunde**, jeweils
links/rechts verschachtelt, als **signed PCM 16 Bit little endian** (192.000 Bytes
pro Sekunde). Der Stream enthält **keinen RIFF-/WAV-Header** und keine Paketheader.
TCP-Lesegrenzen sind keine Audio-Framegrenzen. Das entspricht genau:

```sh
ffmpeg -i Musik.mp3 -f s16le -acodec pcm_s16le -ar 48000 -ac 2 - | nc 192.168.4.1 4041
```

Die WAV-Quelldateien werden vor der Übertragung dekodiert und auf dieses Format
gebracht. Ein WAV-Header auf der TCP-Verbindung würde zum vorhandenen Empfänger
nicht passen.

## Verarbeitung

- OpenAL rendert das räumliche Audio mit HRTF über `ALC_SOFT_loopback` direkt in
  Stereo-PCM; es wird dabei nicht zusätzlich lokal abgespielt.
- Android TTS synthetisiert im Netzwerkmodus in temporäre WAV-Dateien. Nach der
  Synthese werden sie in die Sprachwarteschlange des Mixers übernommen und
  gelöscht. Fertigmeldungen erfolgen nach dem Abspielen im Mixer, sodass die
  Spracherkennung nicht bereits nach der Synthese neu startet. Lange einzelne
  Äußerungen verursachen daher eine zusätzliche Wartezeit bis zum Sprachbeginn.
- `AppSoundPlayer` führt Signaltöne und Tutorials demselben Mixer zu. Pause und
  Fortsetzen der Tutorials erhalten die Abspielposition.
- `PcmMixer` mischt gleichzeitig aktive Quellen mit Begrenzung auf den gültigen
  16-Bit-Bereich. Quellen derselben Sprachwarteschlange laufen nacheinander.
- Alle 10 ms entsteht ein Audioblock. Der Netzwerksender puffert höchstens fünf
  Blöcke. Bei Verbindungsproblemen werden alte Blöcke verworfen, statt Audio
  minutenlang nachzuspielen. Verbindungen werden automatisch erneut aufgebaut;
  ein länger als eine Sekunde blockierter Schreibzugriff wird abgebrochen.

Die Übertragung benötigt keine Bildschirmaufnahme oder MediaProjection-Freigabe.
Neue Audioquellen müssen über `AppSoundPlayer` bzw. `AppAudioOutput.destination()`
eingespeist werden, damit sie Teil des gemeinsamen Streams sind.

## Prüfung

`PcmMixerTest` prüft Mischung und Übersteuerungsbegrenzung, Stereo-Kanalreihenfolge,
Sprachwarteschlange, Abbruch, Tutorial-Pause, WAV-Metadaten, Resampling und die fünf
vorhandenen Signal-/Tutorial-Dateien.
`PcmNetworkStreamTest` prüft an echten lokalen TCP-Verbindungen die unveränderten
PCM-Bytes ohne Header, das Schließen der Verbindung und den Wiederaufbau nach
einem Verbindungsabbruch.

Für einen vollständigen Build zuerst im Verzeichnis `eye-ai-core-rs`
`cargo build-android` ausführen. Dadurch entstehen die aktualisierte native
Bibliothek und die UniFFI-Kotlin-Bindings. Danach in `EyeAIApp`
`gradlew.bat :app:testDebugUnitTest :app:assembleDebug` ausführen.

Auf einem Android-Gerät mit erreichbarem EyeAIVision prüfen:

1. EyeAIVision auswählen und den Betrieb starten; Ton kommt am Zielgerät an.
2. TTS, Objekt-/Tiefenaudio, Aktivierungs-/Deaktivierungston und Tutorials prüfen.
3. Tutorial pausieren und fortsetzen; TTS stoppen bzw. durch neue Sprache ersetzen.
4. WLAN trennen und wieder verbinden; aktuelle Töne kommen nach der erneuten
   Verbindung an, ohne ältere Äußerungen nachzuspielen.
5. Eingabequelle wechseln bzw. Betrieb stoppen; TCP-Verbindung endet und die
   lokale Audioausgabe funktioniert wieder.
