import threading
import time
import os

import cv2

class CameraThread(threading.Thread):
    def __init__(self, name, width, height, output_dir, capture_delay):
        super().__init__()
        self.cam = cv2.VideoCapture(0)
        self.frame = None
        self.is_running = True
        self.name = name
        self.width = width
        self.height = height
        self.capture_delay = capture_delay

        self.save_frames = True
        self.image_count = 0

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # setup output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        print("[Camera] Kameraprozess gestartet!")

        while self.is_running:
            ret, raw_frame = self.cam.read()
            _, self.frame = cv2.imencode('.jpg', raw_frame)

            if self.save_frame:
                with open(self.output_dir / f"{self.image_count}.jpg", "wb") as file:
                    file.write(self.frame.tobytes())
                self.image_count += 1

            time.sleep(self.capture_delay)

        self.cam.release()
        print("[Camera] Kameraprozess beendet!")

    def save_frame(self, jpeg):
        pass
    