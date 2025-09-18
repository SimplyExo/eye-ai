# EyeAIServer

Dies ist ein für Testzwecke erstelltes Programm zum emulieren einer EyeAI-Datenbrille

## Nutzung

Erstellen eines Virtual Environments mit allen Dependencies:

```bash
python3 -m venv venv/
source venv/bin/activate
pip3 install opencv-python flask flasgger
sudo ../venv/bin/python3 main.py # Root ist wichtig!
```

> [!note]
> Vor dem Starten der Programme immer erst mit `source venv/bin/activate` das Virtual Environment aktivieren!

> [!note]
> Das Programm muss immer mit root gestartet werden, da Port 80 sonst nicht zugänglich ist!

## API-Docs
Im Browser den Link ````http://localhost:8888/apidocs```` öffnen

