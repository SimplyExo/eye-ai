import ConfigManager

from flask import Flask, Response, jsonify
from flask_socketio import SocketIO
from pathlib import Path
import shutil

from CameraThread import CameraThread

app = Flask(__name__)
socketio = SocketIO(app)
config = ConfigManager.ConfigManager()
cameraThread = CameraThread(
    "Cam", 
    config.get_width(), 
    config.get_height(), 
    config.get_outputdir(), 
    config.get_capturedelay()
)

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

    return jsonify(get_stats())

@app.route("/stats")
def stats():
    return get_stats()

def send_stats():
    while True:
        socketio.emit("stats_update", get_stats())
        socketio.sleep(1)

def gather_img():
    while True:
        frame = cameraThread.get_frame()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n'

def get_stats():
    stats = {
        "recording": cameraThread.save_frames,
        "image_dir": str(cameraThread.output_dir.absolute()),
        "images_taken": cameraThread.image_count,
        "storage_left": free_space_formatted()
    }

    return stats

def free_space_formatted():
    total, used, free = shutil.disk_usage(cameraThread.output_dir.absolute())
    return f"{free // (2**30)} GiB"

def main():
    # start camera thread
    cameraThread.start()

    socketio.start_background_task(send_stats)
    socketio.run(app)

    app.run()

if __name__ == "__main__":
    main()
