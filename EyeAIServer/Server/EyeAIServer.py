import socket
import threading
import time
from threading import Thread
import ssl

import cv2
import json

from flasgger import Swagger
from flask import Flask, Response, request

from Camera import Camera
from ButtonThread import ButtonThread


class EyeAIServer:
    def __init__(self, cam: Camera, port: int, fps: int, use_https: bool = False, cert_path: str = None, key_path: str = None):
        super().__init__()
        self.div_content = ""
        self.address = None     # current client
        self.cam = cam
        self.port = port
        self.fps = fps
        self.use_https = use_https
        self.cert_path = cert_path
        self.key_path = key_path

        self.app = Flask(__name__)

        # Flasgger OpenAPI template
        template = {
            "swagger": "2.0",
            "info": {
                "title": "EyeAI Server API",
                "description": "API Dokumentation für den EyeAI Server",
                "version": "1.0.0"
            },
            "basePath": "/",
            "schemes": [
                "http",
                "https"
            ],
        }
        self.swagger = Swagger(self.app, template=template)

        # Starte Listener Thread
        t = threading.Thread(target=self.listener_thread)
        t.daemon = True
        t.start()


    def start(self):
        @self.app.route("/")
        def index():
            
            protocol = "https" if self.use_https else "http"
            return f"""
            <body style="background: black;">
                <div style="width: 240px; margin: 0px auto;">
                    <img src="/cam0" />
                    <img src="/cam1" />
                </div>
                <div style="color: white; text-align: center; margin-top: 20px;">
                    Server läuft auf {protocol.upper()} Port {self.port}
                </div>
            </body>
            """

        @self.app.route("/cam0")
        def cam():
            """
            Video-Stream von Kamera
            ---
            responses:
              200:
                description: Multipart MJPEG Stream
                content:
                  multipart/x-mixed-replace; boundary=frame:
                    schema:
                      type: string
                      format: binary
            """
            return Response(self.gather_img(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route("/set_fps")
        def set_fps():
            """
            Setzt die Frames per Second (fps) für den Kamera-Stream
            ---
            parameters:
              - name: fps
                in: query
                type: integer
                required: true
                description: Die gewünschte FPS-Zahl
            responses:
              200:
                description: Bestätigung der gesetzten FPS
                examples:
                  text: "Framerate set to 30 fps"
            """
            self.fps = int(request.args.get('fps'))
            return f"Framerate set to {self.fps} fps"

        
        if self.use_https and self.cert_path and self.key_path:
            # HTTPS with self-signed certificate
            print(f"[SERVER] Starte HTTPS Server auf Port {self.port}")
            print(f"[SERVER] Verwende Zertifikat: {self.cert_path}")
            print(f"[SERVER] Verwende Key: {self.key_path}")
            
            # SSL context
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.cert_path, self.key_path)
            
            thread = Thread(target=lambda: self.app.run(
                host='0.0.0.0', 
                port=self.port, 
                threaded=True, 
                debug=False, 
                use_reloader=False,
                ssl_context=context
            ))
        else:
            # HTTP without ssl
            print(f"[SERVER] Starte HTTP Server auf Port {self.port}")
            thread = Thread(target=lambda: self.app.run(
                host='0.0.0.0', 
                port=self.port, 
                threaded=True, 
                debug=False, 
                use_reloader=False
            ))
            
        thread.daemon = True
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

        # only one client supported
        while True:
            connection, address = self.server_socket.accept()

            try:
                print(f"[BUTTON] Client {address} verbunden!")
                while True:
                    connection.send(b'\xff')
                    time.sleep(0.1)

            except:  
                print(f"[BUTTON] Client {address} getrennt!")
