import sys

import cv2

from CameraClient import CameraClient

if len(sys.argv) < 2:
    print("FEHLER: Bitte geben Sie die IP-Adresse des Servers ein!\nProgramm wird beendet...")
    exit(0)


client1 = CameraClient(sys.argv[1], 3333, 'Client 1') # Kamera links
client2 = CameraClient(sys.argv[1], 3334, 'Client 2') # Kamera rechts

try:
    client1.start()
    client2.start()

    print("Clients wurden gestartet")

    while client1.is_running and client2.is_running:
        if client1.frame is not None:
            cv2.imshow(client1.window_name, client1.frame)

        if client2.frame is not None:
            cv2.imshow(client2.window_name, client2.frame)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            client1.disconnect()

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            client2.disconnect()

except:
    print("Programm wird beendet")
    client1.disconnect()
    client2.disconnect()
    exit(0)
