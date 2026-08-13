import time

from flask import Flask, Response
from pathlib import Path

from CameraThread import CameraThread

app = Flask(__name__)
cameraThread = CameraThread("Cam", 1280, 720, Path("./output"), 3)

@app.route("/")
def index():
    with open("html/index.html", "r") as f:
        return f.read()

@app.route("/cam")
def camera():
    return Response(gather_img(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/toggle_recording")
def toggle_recording():
    cameraThread.save_frames = not cameraThread.save_frames

    if cameraThread.save_frames:
        return "Started recording!"
    else:
        return "Stopped recording!"

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
