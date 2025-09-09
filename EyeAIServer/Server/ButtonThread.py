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
                self.connection.send(b'\xff\n')
                time.sleep(0.1)

        except:     # Wenn Client sich trennt
            print(f"[BUTTON] Client {self.address} getrennt!")
            return
