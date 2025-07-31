import socket
import time
from threading import Thread

import cv2
from flask import Flask, Response

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

        self.app = Flask(__name__)


    def start(self):
        @self.app.route("/")
        def hello_world():
            return """
            <body style="background: black;">
                <div style="width: 240px; margin: 0px auto;">
                    <img src="/cam0" />
                </div>
            </body>
            """  # setup camera and resolution

        @self.app.route("/cam0")
        def cam0():
            return Response(self.gather_img(), mimetype='multipart/x-mixed-replace; boundary=frame')

        thread = Thread(target=lambda: self.app.run(host='0.0.0.0', port=self.port, threaded=True))
        thread.start()

    def gather_img(self):
        while True:
            time.sleep(1.0 / self.fps)
            _, frame = cv2.imencode('.jpg', self.cam.frame)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n'