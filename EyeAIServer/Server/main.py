from Camera import Camera
from EyeAIServer import EyeAIServer
import os


def main():
    camera_thread = Camera("Kamera", 640, 640)
    camera_thread.start()

    tcp_port = 3333
    web_port = 80

    use_https = False  # for https
    cert_path = "cert.pem"  # path to certificate
    key_path = "key.pem"   # path to private key
    
    
    if use_https:
        if not os.path.exists(cert_path):
            print(f"[ERROR] Zertifikat nicht gefunden: {cert_path}")
            print("[INFO] Wechsle zu HTTP...")
            use_https = False
        elif not os.path.exists(key_path):
            print(f"[ERROR] Private Key nicht gefunden: {key_path}")
            print("[INFO] Wechsle zu HTTP...")
            use_https = False
    
    
    c1 = EyeAIServer(
        camera_thread, 
        web_port,
        tcp_port,
        30,
        use_https=use_https,
        cert_path=cert_path if use_https else None,
        key_path=key_path if use_https else None
    )
    c1.start()
    
    protocol = "HTTPS" if use_https else "HTTP"
    print(f"\n[INFO] Server läuft auf {protocol}://localhost:{web_port}")
    print(f"[INFO] Erreichbar im Netzwerk über {protocol}://<SERVER-IP>:{web_port}")
    
    if use_https:
        print("\n[WARNUNG] Self-signed Zertifikat wird verwendet!")
        print("[INFO] Browser werden eine Sicherheitswarnung anzeigen.")
        print("[INFO] Die Android App sollte mit trustAllCertificates=true funktionieren.\n")
    
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Server wird beendet...")

if __name__ == "__main__":
    main()
