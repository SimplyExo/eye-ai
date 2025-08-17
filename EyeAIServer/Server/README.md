\page EyeAIServer_Server Server

## Nutzung

#### 1. Aktivieren des zuvor erstellten Virtual Environments:

```
source ../venv/bin/activate
```

#### 2. Starten des Servers

```
python3 main.py
```

Dadurch wird ein Server gestartet, welcher die Webcam des PCs nutzt, um eine Stereokamera zu "simulieren"
(es wird ein und dasselbe Bild auf zwei verschiedenen Ports an den Client gesendet)

Zur besseren Unterscheidung der beiden Kanäle wird einer von diesen auf einer geringen Framerate (hier 3 FPS)
übertragen
