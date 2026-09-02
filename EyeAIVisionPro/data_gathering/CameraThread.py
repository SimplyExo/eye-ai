import threading
import time
import os
import cv2
from datetime import datetime

from ConfigManager import ConfigManager

class CameraThread(threading.Thread):
    def __init__(self, name, config: ConfigManager):
        super().__init__()
        self.cam = cv2.VideoCapture(0)
        self.config = config
        self.frame = None
        self.is_running = True
        self.name = name

        self.image_ready = False        # TODO: find better solution for this
        self.save_frames = False
        self.image_count = 0
        self.fps = 30

        self.next_image_time = 0

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.get_width())
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get_height())

        # setup output directory
        os.makedirs(config.get_outputdir(), exist_ok=True)

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

    def get_taken_images(self):
        i = 0
        for file in os.listdir(self.config.get_outputdir().absolute()):
            if file.endswith(".jpeg") or file.endswith(".jpg"):
                i += 1

        return i

    def get_frame(self):
        while not self.image_ready:
            pass

        self.image_ready = False
        return self.frame

    def save_frame(self):
        self.next_image_time = time.time() + self.config.get_capturedelay()
        new_filename = f"{datetime.now().strftime("%d%m%y_%H%M%S.%f")}.jpg"
        with open(self.config.get_outputdir() / new_filename, "wb") as file:
            file.write(self.frame.tobytes())
        self.image_count += 1
    