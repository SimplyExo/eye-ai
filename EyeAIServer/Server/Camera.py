import threading

import cv2


class Camera(threading.Thread):
    def __init__(self, name):
        super().__init__()
        self.cam = cv2.VideoCapture(0)
        self.frame = None
        self.is_running = True
        self.name = name

    def run(self):
        print("[Camera] Kameraprozess gestartet!")

        while self.is_running:
            ret, self.frame = self.cam.read()

            # Display the captured frame
            cv2.imshow(self.name, self.frame)

            # Press 'q' to exit the program
            if cv2.waitKey(1) == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()
        print("[Camera] Kameraprozess beendet!")
