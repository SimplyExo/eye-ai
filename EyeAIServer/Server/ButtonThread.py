import threading
import time


class ButtonThread(threading.Thread):
    def __init__(self, connection, address):
        super().__init__()
        self.connection = connection
        self.address = address

    def run(self):
        try:
            print(f"[BUTTON] Client {self.address} verbunden!")
            while True:
                byte_to_send = input("[BUTTON] Bitte Clickzahl eingeben (1 oder 2): ")

                if byte_to_send == "1" or byte_to_send == "2":
                    self.connection.send(byte_to_send.encode())
                else:
                    print("[BUTTON] Dies ist keine zulässige Eingabe!")

        except:     # Wenn Client sich trennt
            print(f"[BUTTON] Client {self.address} getrennt!")
            return
