import numpy as np
import time
from math import sqrt

def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

class SampleCollector:
    def __init__(self, chessboard_size=None, square_size=None):
        self.objp = None
        if chessboard_size and square_size:
            self.objp = np.zeros(
                (chessboard_size[0] * chessboard_size[1], 3),
                np.float32
            )
            self.objp[:, :2] = np.mgrid[
                0:chessboard_size[0],
                0:chessboard_size[1]
            ].T.reshape(-1, 2)
            self.objp *= square_size

        self.objpoints = []
        self.imgpoints = []
        self.charuco_corners = []
        self.charuco_ids = []
        self.last_capture_time = 0
        self.last_center = None

    def add_sample(self, corners, min_center_shift=None, capture_delay=None, charuco_ids=None):
        center_x = np.mean(corners[:, 0, 0])
        center_y = np.mean(corners[:, 0, 1])
        current_center = (center_x, center_y)

        current_time = time.time()
        status_text = ""
        allow_capture = True

        if self.last_center is not None and min_center_shift is not None:
            shift = distance(current_center, self.last_center)
            if shift < min_center_shift:
                allow_capture = False
                status_text = f"Move board more ({shift:.1f}px)"

        if capture_delay is not None and current_time - self.last_capture_time < capture_delay:
            allow_capture = False
            if not status_text:
                status_text = "Waiting..."
        
        if allow_capture:
            if charuco_ids is not None:
                self.charuco_corners.append(corners)
                self.charuco_ids.append(charuco_ids)
            else:
                self.objpoints.append(self.objp.copy())
                self.imgpoints.append(corners)
            
            self.last_capture_time = current_time
            self.last_center = current_center
            status_text = f"Captured {self.get_sample_count()}"
            print(status_text)
            return True, status_text

        return False, status_text

    def get_points(self):
        if self.charuco_corners:
            return self.charuco_corners, self.charuco_ids
        return self.objpoints, self.imgpoints

    def get_sample_count(self):
        if self.charuco_corners:
            return len(self.charuco_corners)
        return len(self.imgpoints)
