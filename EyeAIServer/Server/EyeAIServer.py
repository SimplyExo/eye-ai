import socket
import threading
import time
from threading import Thread

import cv2
import json
from flask import Flask, Response, request

from Camera import Camera
from Server.ButtonThread import ButtonThread


class EyeAIServer:
    def __init__(self, cam: Camera, port: int, fps: int):
        super().__init__()
        self.div_content = ""
        self.address = None     # Aktuell verbundener Client
        self.cam = cam
        self.port = port
        self.fps = fps

        self.known_networks = {     # Beispielwerte
              "networks": [
                {"ssid": "123", "password": "ha"},
                {"ssid": "456", "password": "ha1"},
                {"ssid": "789", "password": "ha2"}
              ]
            }

        self.app = Flask(__name__)

        t = threading.Thread(target=self.listener_thread)
        t.start()


    def start(self):
        @self.app.route("/")
        def index():
            return f"""
            <body style="background: black;">
                <div style="width: 240px; margin: 0px auto;">
                    <img src="/cam0" />
                    <img src="/cam1" />
                </div>
            </body>
            """  # setup cameras and resolution

        @self.app.route("/cam0")
        @self.app.route("/cam1")
        def cam():
            return Response(self.gather_img(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route("/set_fps")
        def set_fps():
            self.fps = int(request.args.get('fps'))
            return f"Response: {self.fps} fps"

        @self.app.route("/set_networks", methods=['POST'])
        def set_networks():
            content = request.json

            if "networks" in content:
                for net in content["networks"]:
                    if not "ssid" in net and not "password" in net:
                        return "Bad request, check your JSON!", 400

            else:
                return "Bad request, check your JSON!", 400

            self.known_networks = content
            return "OK"

        @self.app.route("/get_networks", methods=['GET'])
        def get_networks():
            return self.app.response_class(
                response=json.dumps(self.known_networks),
                mimetype='application/json'
            )

        @self.app.route("/add_networks", methods=['POST'])
        def add_networks():
            content = request.json
            response = ""

            if "networks" in content:
                for net in content["networks"]:
                    if "ssid" in net and "password" in net:
                        self.known_networks["networks"].append(net)
                        response += f"Added: {net}\n"

                    else:
                        return "Bad request, check your JSON!", 400

            else:
                return "Bad request, check your JSON!", 400

            return response

        @self.app.route("/del_networks", methods=['POST'])
        def del_networks():
            content = request.json
            response = ""

            if "networks" in content:
                for net in content["networks"]:
                    if "ssid" in net and "password" in net:
                        try:
                            self.known_networks["networks"].remove(net)
                            response += f"Removed: {net}\n"

                        except ValueError:
                            response += f"Failed to remove: {net}\n"

                    else:
                        return "Bad request, check your JSON!", 400

            else:
                return "Bad request, check your JSON!", 400

            return response

        thread = Thread(target=lambda: self.app.run(host='0.0.0.0', port=self.port, threaded=True))
        thread.start()

    def gather_img(self):
        while True:
            time.sleep(1.0 / self.fps)
            _, frame = cv2.imencode('.jpg', self.cam.frame)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n'

    def listener_thread(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', 8080))
        self.server_socket.listen(1)

        # Nur ein Client darf sich verbinden, daher kein Multithreading!
        while True:
            connection, address = self.server_socket.accept()

            try:
                print(f"[BUTTON] Client {address} verbunden!")
                while True:
                    connection.send(b'\xff')
                    time.sleep(0.1)

            except:  # Wenn Client sich trennt
                print(f"[BUTTON] Client {address} getrennt!")
