import cv2
import sys
import time


class Camera:
    def __init__(self, cam_id=0, width=640, height=480, fps=30):

        # =====================================================
        # Backend selection (CRITICAL for Windows stability)
        # =====================================================
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY

        self.cap = cv2.VideoCapture(cam_id + backend)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_id}")

        # =====================================================
        # Basic configuration
        # =====================================================
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # =====================================================
        # Stability improvements (VERY IMPORTANT)
        # =====================================================
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # warm-up camera (removes initial jitter)
        for _ in range(10):
            self.cap.read()

    # =========================================================
    def read(self):
        """
        Returns latest frame with buffer drop (prevents lag/jitter)
        """

        # drop old frames (CRITICAL for USB webcams)
        for _ in range(2):
            self.cap.grab()

        ret, frame = self.cap.read()
        return ret, frame

    # =========================================================
    def release(self):
        self.cap.release()