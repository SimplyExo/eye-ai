from Camera import Camera
from EyeAIServer import EyeAIServer
import os


def main():
    camera_thread = Camera("Kamera", 640, 640)
    camera_thread.start()

    # HTTPS Konfiguration
    use_https = True  # Auf True setzen für HTTPS
    cert_path = "cert.pem"  # Pfad zum Zertifikat
    key_path = "key.pem"   # Pfad zum Private Key
    
    # Prüfe ob Zertifikate existieren
    if use_https:
        if not os.path.exists(cert_path):
            print(f"[ERROR] Zertifikat nicht gefunden: {cert_path}")
            print("[INFO] Wechsle zu HTTP...")
            use_https = False
        elif not os.path.exists(key_path):
            print(f"[ERROR] Private Key nicht gefunden: {key_path}")
            print("[INFO] Wechsle zu HTTP...")
            use_https = False
    
    # Server starten (Port 3333 für HTTPS oder HTTP)
    c1 = EyeAIServer(
        camera_thread, 
        3333,  # Du kannst auch 443 für Standard HTTPS verwenden
        30,
        use_https=use_https,
        cert_path=cert_path if use_https else None,
        key_path=key_path if use_https else None
    )
    c1.start()
    
    protocol = "HTTPS" if use_https else "HTTP"
    print(f"\n[INFO] Server läuft auf {protocol}://localhost:{3333}")
    print(f"[INFO] Erreichbar im Netzwerk über {protocol}://<SERVER-IP>:{3333}")
    
    if use_https:
        print("\n[WARNUNG] Self-signed Zertifikat wird verwendet!")
        print("[INFO] Browser werden eine Sicherheitswarnung anzeigen.")
        print("[INFO] Die Android App sollte mit trustAllCertificates=true funktionieren.\n")
    
    # Server läuft im Hintergrund, main Thread am Leben halten
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Server wird beendet...")

if __name__ == "__main__":
    main()
