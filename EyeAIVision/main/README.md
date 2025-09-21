# EyeAIVision

### Software
Das Programm der EyeAIVision wurde mithilfe von ESP-IDF in C geschrieben, dem Entwicklungsframework des Herstellers Espressif. Zwar ist es auch möglich, den ESP32 in der unter Hobbyentwicklern weiter verbreiteten Arduino IDE zu programmieren, allerdings sind einige Features des Mikrocontrollers einfacher in ESP-IDF zu handhaben. Im Gegensatz zur Super Loop der Arduino IDE ist es in ESP-IDF möglich, ein Real Time Operating System (hier FreeRTOS) zu verwenden, welches das Management von verschiedenen Tasks unterstützt.

Beim Einschalten des Geräts beginnt das Programm zunächst mit dem Initialisieren von Pins der beiden LED-Farben sowie dem Touch-Pin. Hierbei wird die Farbe der Status-LED auf Rot gesetzt, da das Gerät noch keine Verbindung hergestellt hat. Anschließend wird die Kamera initialisiert, die in diesem Schritt auf die gewünschte Auflösung von 640x480 eingestellt wird.

Darauf folgt das Starten von WiFi im Station-Mode. Hierbei agiert die EyeAIVision wie ein Client und stellt kein eigenes Netzwerk her. Damit sich das Smartphone mit der Brille verbinden kann, muss der mobile Hotspot auf diesem aktiviert sein und als SSID „EyeAI“ sowie als Passwort „123456789“ konfiguriert werden. Aufgrund von unterschiedlichen Sicherheitseinschränkungen war es nicht möglich, innerhalb von EyeAIApp ein eigenes Netzwerk herzustellen. Auch ein vom Mikrocontroller ausgehendes Netzwerk erwies sich als ungeeignet, da es die Übertragungsgeschwindigkeit der Kameraframes massiv bremste.

Solange die EyeAIVision nicht mit dem Netzwerk verbunden ist, versucht sie kontinuierlich, eine Verbindung aufzubauen. Sobald ein geeigneter Hotspot gefunden und verbunden ist, wird die Status-LED auf Grün geschaltet und EyeAIApp kann gestartet werden.

Daraufhin wird ein TCP-Server auf Port 3333 gestartet. Dieser ist dazu da, den Status des Touch-Buttons an die App zu senden. Wird ein Einzelklick durchgeführt, so wird „1“ übertragen, bei einem Doppelklick die „2“. Diese Signale werden anschließend in EyeAIApp interpretiert.

Zum Schluss wird der HTTP-Server mit dem MJPEG-Stream gestartet. Unter „http://<ip-des-geräts>/cam0“ kann dann der Stream ausgelesen werden.

### Installation
1. ESP-IDF v5.5 installieren (siehe https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32/get-started/index.html)
2. Anschließen des ESP32 CAM mit dem Programmerboard (oder einem FTDI Programmer) per USB
3. Kompilieren des Codes mit ```idf.py build```
4. Flashen des Programms mit ```idf.py -p /dev/<Gerät> flash```
5. Optional: Anzeigen der Ausgabe des Controllers mit ```idf.py -p monitor```

> **Hinweis:** Stellen Sie sicher, dass esp-idf in der aktuellen Terminal-Session aktiviert ist!

> **Hinweis:** Stelle Sie sicher, dass der Hotspot auf dem Smartphone mit folgenden Credentials aktiviert ist, bevor du die Brille einschaltest:
> SSID: EyeAI, Passwort: 123456789
