import threading
import time
import socket

import cv2
import numpy as np


class CameraClient(threading.Thread):
    REQUEST_BYTE = b'\xFF'
    MAX_PACKET_SIZE = 1030  # 2 (SEQ) + 1 (END) + 1024 DATA
    is_running = False
    frame = None

    def __init__(self, ip: str, port: int, window_name: str, show_fps = False):
        super().__init__()
        self.UDP_IP = ip
        self.UDP_PORT = port
        self.show_fps = show_fps
        self.window_name = window_name

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, 0))

    def connect(self):
        self.is_running = True
        self.sock.sendto(self.REQUEST_BYTE, (self.UDP_IP, self.UDP_PORT))

    def disconnect(self):
        self.is_running = False
        self.sock.sendto(self.REQUEST_BYTE, (self.UDP_IP, self.UDP_PORT))

    def run(self):
        self.connect()

        while self.is_running:
            # Buffer für empfangene Daten
            data_parts = {}
            done = False

            start = time.time()

            while not done:
                packet, addr = self.sock.recvfrom(self.MAX_PACKET_SIZE)
                seq = int.from_bytes(packet[0:2], 'big')
                end_flag = packet[2]
                chunk_data = packet[3:]

                data_parts[seq] = chunk_data
                if end_flag == 1:
                    done = True

            # Pakete in der richtigen Reihenfolge zusammensetzen
            jpeg_data = b''.join(data_parts[i] for i in sorted(data_parts))
            nparr = np.frombuffer(jpeg_data, np.uint8)
            self.frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            elapsed = time.time() - start

            if self.show_fps:
                print(1.0 / elapsed)
