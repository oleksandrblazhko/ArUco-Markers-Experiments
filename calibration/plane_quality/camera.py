import cv2
import sys

class CameraManager:
    def __init__(self, cam_index, width, height):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cam_index, backend)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {cam_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self.cap.release()
