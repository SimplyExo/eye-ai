import time

from flask import Flask, Response
from pathlib import Path

from CameraThread import CameraThread

app = Flask(__name__)
cameraThread = CameraThread("Cam", 1280, 720, Path("./output"), 3)

@app.route("/")
def index():
    return open("html/index.html", "r").read()

@app.route("/cam")
def camera():
    return Response(gather_img(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gather_img():
    while True:
        frame = cameraThread.frame
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n'

def main():
    # start camera thread
    cameraThread.start()

    app.run()

if __name__ == "__main__":
    main()
