import time

from flask import Flask, Response
from pathlib import Path

from CameraThread import CameraThread

app = Flask(__name__)
cameraThread = CameraThread("Cam", 1280, 720, Path("./output"), 0.5)

@app.route("/")
def index():
    with open("html/index.html", "r") as f:
        return f.read()

@app.route("/cam")
def camera():
    return Response(gather_img(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gather_img():
    while True:
        frame = cameraThread.get_frame()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n'

def main():
    # start camera thread
    cameraThread.start()

    app.run()

if __name__ == "__main__":
    main()
