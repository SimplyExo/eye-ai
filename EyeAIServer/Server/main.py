from Camera import Camera
from CameraServer import CameraServer


def main():
    camera_thread = Camera("Kamera", 640, 640)
    camera_thread.start()

    c1 = CameraServer(camera_thread, 3333, 0, 30)
    c1.start()

    c2 = CameraServer(camera_thread, 3334, 1, 30)
    c2.start()

if __name__ == "__main__":
    main()