import socket
import time
from threading import Thread

import cv2

from Camera import Camera


class CameraServer:
    def __init__(self, cam: Camera, port: int, i: int, fps: int):
        super().__init__()
        self.address = None     # Aktuell verbundener Client
        self.cam = cam
        self.index = i
        self.port = port
        self.fps = fps

        self.CHUNK_SIZE = 1024

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_frame = None

    def start(self):
        rx_thread = Thread(target=self.rx)
        rx_thread.start()

    # Empfänger Thread
    def rx(self):
        self.server_socket.bind(('0.0.0.0', self.port))
        print(f"[Camera Nr. {self.index}] Warte auf Client. Port: {self.port}")

        while True:
            message, address = self.server_socket.recvfrom(1024)
            byte = message[0]

            if byte == 0xff and self.address is None:
                # Thread starten
                self.address = address
                print(f"[Camera Nr. {self.index}] Client {address} angenommen")
                tx_thread = Thread(target=self.tx)
                tx_thread.start()

            elif byte == 0xff and self.address == address:
                self.address = None
                print(f"[Camera Nr. {self.index}] Client {address} wurde ausgeloggt")

            else:
                print(f"[Camera Nr. {self.index}] Client {address} abgelehnt")

    # Sender Thread
    def tx(self):
        print(f"[Camera Nr. {self.index}] Starte Übertragung an {self.address}")

        while self.address is not None:
            frame = self.cam.frame

            # Nur senden, wenn vorheriger und aktueller Frame nicht gleich sind!
            if frame is not None:
                self.last_frame = frame

                # in JPEG kodieren
                success, jpeg_bytes = cv2.imencode('.jpg', frame)
                jpeg_data = jpeg_bytes.tobytes()

                if success:
                    # Aufteilen in Paketen
                    total_packets = (len(jpeg_data) + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
                    for i in range(total_packets):
                        start = i * self.CHUNK_SIZE
                        end = start + self.CHUNK_SIZE
                        chunk = jpeg_data[start:end]
                        # Header: Sequenznummer (2 Bytes), Endekennung (1 Byte)
                        # Format: [SEQ][END_FLAG][DATA]
                        # SEQ: 0–65535, END_FLAG: 1 = letztes Paket, 0 = sonst
                        seq = i.to_bytes(2, 'big')
                        is_last = b'\x01' if (i == total_packets - 1) else b'\x00'
                        packet = seq + is_last + chunk

                        if self.address is not None:
                            self.server_socket.sendto(packet, self.address)

                        else:
                            break

            # Begrenzung der FPS auf Serverseite
            time.sleep(1.0/self.fps)