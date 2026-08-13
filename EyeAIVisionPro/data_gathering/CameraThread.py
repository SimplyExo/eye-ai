import threading
import time

import cv2


class CameraThread(threading.Thread):
    def __init__(self, name, width, height):
        super().__init__()
        self.cam = cv2.VideoCapture(0)
        self.frame = None
        self.is_running = True
        self.name = name
        self.width = width
        self.height = height

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def run(self):
        print("[Camera] Kameraprozess gestartet!")

        while self.is_running:
            ret, self.frame = self.cam.read()
            time.sleep(1.0/30.0)

        self.cam.release()
        print("[Camera] Kameraprozess beendet!")