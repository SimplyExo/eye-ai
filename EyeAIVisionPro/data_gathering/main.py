import ConfigManager

from flask import Flask, Response, jsonify, request
from flask_socketio import SocketIO
from pathlib import Path
import shutil
import os

from CameraThread import CameraThread

app = Flask(__name__)
socketio = SocketIO(app)
config = ConfigManager.ConfigManager()
cameraThread = CameraThread(
    "Cam", 
    config
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

@app.route("/update_settings", methods=["POST"])
def update_settings():
    json_dict = request.json

    try:
        integer_value = int(json_dict["capture_delay"])
    except ValueError:
        return jsonify({"message": f"Error: Cannot set {json_dict["capture_delay"]} as capture delay!"}), 400

    if not os.path.exists(json_dict["output_dir"]):
        return jsonify({"message": f"Error: Cannot set '{json_dict["output_dir"]}' as output directory!"}), 400

    config.set_capturedelay(integer_value)
    config.set_outputdir(Path(json_dict["output_dir"]))
    config.save()

    return jsonify({"message": f"Saved all settings successfully!"})
    

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
        "image_dir": str(cameraThread.config.get_outputdir().absolute()),
        "images_taken": cameraThread.get_taken_images(),
        "storage_left": free_space_formatted(),
        "capture_delay": cameraThread.config.get_capturedelay()
    }

    return stats

def free_space_formatted():
    total, used, free = shutil.disk_usage(cameraThread.config.get_outputdir().absolute())
    return f"{free // (2**30)} GiB"

def main():
    # start camera thread
    cameraThread.start()

    socketio.start_background_task(send_stats)
    socketio.run(app)

    app.run()

if __name__ == "__main__":
    main()
