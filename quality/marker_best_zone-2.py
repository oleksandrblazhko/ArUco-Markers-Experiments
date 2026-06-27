import cv2
import numpy as np
import json
import csv
import math
import time
import os
from dataclasses import dataclass


# ==========================================================
# CONFIGURATION
# ==========================================================

CAMERA_FILE = "camera_ext.json"
EXPERIMENT_FILE = "experiment.json"


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass
class CameraParameters:
    matrix: np.ndarray
    dist: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ExperimentParameters:
    camera_height_mm: float
    frame_width: int
    frame_height: int
    grid_step_mm: float
    grid_width_mm: float
    grid_height_mm: float
    captures_per_point: int
    target_radius_px: int
    aruco_dictionary: str


# ==========================================================
# LOAD CONFIGS
# ==========================================================

def load_camera(filename=CAMERA_FILE):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    with open(filename, "r", encoding="utf8") as f:
        data = json.load(f)

    matrix = np.array(data["mtx"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64)

    return CameraParameters(
        matrix=matrix,
        dist=dist,
        fx=matrix[0, 0],
        fy=matrix[1, 1],
        cx=matrix[0, 2],
        cy=matrix[1, 2]
    )


def load_experiment(filename=EXPERIMENT_FILE):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    with open(filename, "r", encoding="utf8") as f:
        data = json.load(f)

    return ExperimentParameters(
        camera_height_mm=data["camera_height_mm"],
        frame_width=data["frame_width"],
        frame_height=data["frame_height"],
        grid_step_mm=data["grid_step_mm"],
        grid_width_mm=data["grid_width_mm"],
        grid_height_mm=data["grid_height_mm"],
        captures_per_point=data["captures_per_point"],
        target_radius_px=data["target_radius_px"],
        aruco_dictionary=data["aruco_dictionary"]
    )


# ==========================================================
# ARUCO DICTIONARIES
# ==========================================================

ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000
}


# ==========================================================
# GRID
# ==========================================================

def build_grid(exp):
    grid = []

    xmin = -exp.grid_width_mm / 2
    xmax = exp.grid_width_mm / 2
    ymin = -exp.grid_height_mm / 2
    ymax = exp.grid_height_mm / 2

    y = ymin
    while y <= ymax + 1e-6:
        x = xmin
        while x <= xmax + 1e-6:
            grid.append((x, y))
            x += exp.grid_step_mm
        y += exp.grid_step_mm

    return grid


# ==========================================================
# WORLD -> PIXEL
# ==========================================================

def world_to_pixel(X, Y, camera, exp):
    px = camera.fx * X / exp.camera_height_mm + camera.cx
    py = camera.fy * Y / exp.camera_height_mm + camera.cy
    return int(round(px)), int(round(py))


# ==========================================================
# DRAW GRID
# ==========================================================

def draw_grid(image, camera, exp, grid, current):
    for i, (X, Y) in enumerate(grid):
        px, py = world_to_pixel(X, Y, camera, exp)

        color = (180, 180, 180)
        radius = 3

        if i == current:
            color = (0, 0, 255)
            radius = 8

        cv2.circle(image, (px, py), radius, color, -1)


# ==========================================================
# DISTANCE
# ==========================================================

def pixel_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ==========================================================
# MAIN
# ==========================================================

def main():

    camera = load_camera()
    exp = load_experiment()

    grid = build_grid(exp)

    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICTIONARIES[exp.aruco_dictionary]
    )
    detector = cv2.aruco.ArucoDetector(dictionary)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, exp.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, exp.frame_height)

    if not cap.isOpened():
        print("Camera error")
        return

    csvFile = open("result.csv", "w", newline="", encoding="utf8")
    writer = csv.writer(csvFile)

    writer.writerow([
        "Point", "GridX_mm", "GridY_mm",
        "ImageX", "ImageY",
        "DetectionRate",
        "MeanSharpness",
        "StdSharpness",
        "MarkerSize_px",
        "DistanceFromCenter_px"
    ])

    currentPoint = 0

    LOCK_TIME = 0.5
    COUNTDOWN_TIME = 3

    targetLockStart = None
    countdownStart = None

    measuring = False
    message = ""
    messageColor = (0, 255, 0)

    print("Press ESC to exit")

    while True:

        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.undistort(frame, camera.matrix, camera.dist)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        draw_grid(frame, camera, exp, grid, currentPoint)

        corners, ids, _ = detector.detectMarkers(gray)

        markerCenter = None

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            c = corners[0][0]
            markerCenter = (
                int(np.mean(c[:, 0])),
                int(np.mean(c[:, 1]))
            )

            cv2.circle(frame, markerCenter, 5, (255, 0, 0), -1)

        targetX, targetY = grid[currentPoint]
        targetPixel = world_to_pixel(targetX, targetY, camera, exp)

        if markerCenter is not None:

            dist = pixel_distance(markerCenter, targetPixel)

            if dist <= exp.target_radius_px:

                cv2.circle(frame, targetPixel, 10, (0, 255, 0), 2)

                if targetLockStart is None:
                    targetLockStart = time.time()

                lock_time = time.time() - targetLockStart

                if lock_time >= LOCK_TIME:

                    if countdownStart is None:
                        countdownStart = time.time()

                    elapsed = time.time() - countdownStart
                    sec = COUNTDOWN_TIME - int(elapsed)

                    if sec > 0:
                        message = str(sec)
                        messageColor = (0, 255, 0)
                    else:
                        measuring = True
                else:
                    message = f"HOLD {LOCK_TIME - lock_time:.1f}s"
                    messageColor = (0, 255, 255)

            else:
                targetLockStart = None
                countdownStart = None
                measuring = False
                message = "MOVE MARKER"
                messageColor = (0, 0, 255)

        cv2.putText(frame, f"Point {currentPoint+1}/{len(grid)}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, message,
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, messageColor, 2)

        # ======================================================
        # MEASUREMENT
        # ======================================================

        if measuring:

            detected = 0
            sharpnessValues = []
            markerSizes = []

            for i in range(exp.captures_per_point):

                ok, img = cap.read()
                if not ok:
                    continue

                img = cv2.undistort(img, camera.matrix, camera.dist)
                gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                c2, id2, _ = detector.detectMarkers(gray2)

                if id2 is None:
                    continue

                detected += 1

                cc = c2[0][0]

                side = (
                    np.linalg.norm(cc[0] - cc[1]) +
                    np.linalg.norm(cc[1] - cc[2]) +
                    np.linalg.norm(cc[2] - cc[3]) +
                    np.linalg.norm(cc[3] - cc[0])
                ) / 4.0

                markerSizes.append(side)

                xmin, xmax = int(np.min(cc[:, 0])), int(np.max(cc[:, 0]))
                ymin, ymax = int(np.min(cc[:, 1])), int(np.max(cc[:, 1]))

                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax = min(gray2.shape[1] - 1, xmax)
                ymax = min(gray2.shape[0] - 1, ymax)

                if xmin >= xmax or ymin >= ymax:
                    continue

                roi = gray2[ymin:ymax, xmin:xmax]

                if roi.size > 0:
                    sharp = cv2.Laplacian(roi, cv2.CV_64F).var()
                    sharpnessValues.append(sharp)

            detectionRate = 100 * detected / exp.captures_per_point
            meanSharp = np.mean(sharpnessValues) if sharpnessValues else 0
            stdSharp = np.std(sharpnessValues) if sharpnessValues else 0
            meanSize = np.mean(markerSizes) if markerSizes else 0

            distCenter = pixel_distance(targetPixel, (camera.cx, camera.cy))

            writer.writerow([
                currentPoint + 1,
                targetX, targetY,
                targetPixel[0], targetPixel[1],
                round(detectionRate, 2),
                round(meanSharp, 2),
                round(stdSharp, 2),
                round(meanSize, 2),
                round(distCenter, 2)
            ])

            csvFile.flush()

            print(f"Point {currentPoint+1}: detection {detectionRate:.2f}%")

            currentPoint += 1
            measuring = False
            targetLockStart = None
            countdownStart = None

            if currentPoint >= len(grid):
                print("Experiment finished")
                break

        cv2.imshow("ArUco Experiment", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    csvFile.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()