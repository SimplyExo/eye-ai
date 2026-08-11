import time

import cv2
from flask import Flask, Response

app = Flask(__name__)
cam = cv2.VideoCapture(0)

@app.route("/")
def index():
    return open("html/index.html", "r").read()

@app.route("/cam")
def camera():
    return Response(gather_img(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gather_img():
    while True:
        #time.sleep(1.0 / 30.0)
        ret, raw_frame = cam.read()
        _, frame = cv2.imencode('.jpg', raw_frame)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n'

def main():
    # setup cv2 camera
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    app.run()

if __name__ == "__main__":
    main()
