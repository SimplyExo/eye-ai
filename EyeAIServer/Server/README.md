# EyeAIServer
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
(es wird dasselbe Bild per HTTP auf zwei verschiedenen Ports an den Client gesendet)

#### 3. Anzeigen des Bildes
Zum Anzeigen müssen folgende Links im Browser geöffnet werden:\
Kamera 1:
```
http://localhost:3333/cam0
```
Kamera 2:
```
http://localhost:3334/cam0
```
