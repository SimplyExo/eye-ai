from Camera import Camera
from EyeAIServer import EyeAIServer


def main():
    camera_thread = Camera("Kamera", 640, 640)
    camera_thread.start()

    c1 = EyeAIServer(camera_thread, 3333, 30)
    c1.start()

if __name__ == "__main__":
    main()