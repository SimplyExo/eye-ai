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

        self.image_ready = False        # TODO: find better solution for this
        self.save_frames = False
        self.image_count = 0
        self.fps = 30

        self.next_image_time = time.time()

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
            self.image_ready = True

            if self.save_frames and time.time() >= self.next_image_time:
                self.save_frame()

            time.sleep(1 / self.fps)

        self.cam.release()
        print("[Camera] Kameraprozess beendet!")

    def get_frame(self):
        while not self.image_ready:
            pass

        self.image_ready = False
        return self.frame

    def save_frame(self):
        self.next_image_time += self.capture_delay
        with open(self.output_dir / f"{self.image_count}.jpg", "wb") as file:
            file.write(self.frame.tobytes())
        self.image_count += 1
    