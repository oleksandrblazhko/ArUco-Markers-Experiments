import argparse
from enum import Enum

import cv2

# ==========================================================
# Program Mode
# ==========================================================

class ProgramMode(Enum):
    CALIBRATION = "calibration"
    PLANE_QUALITY = "plane_quality"


# ==========================================================
# Plane Quality (default constants)
# ==========================================================

PLANE_GRID_ROWS = 6
PLANE_GRID_COLS = 8

PLANE_MIN_SAMPLES = 30
PLANE_ANGLE_THRESHOLD = 2.5
PLANE_DISTANCE_THRESHOLD = 0.01

PLANE_HEATMAP_RESOLUTION = 200
PLANE_OUTPUT_DIR = "results"

DEFAULT_PLANE_FRAMES = 200
DEFAULT_MIN_MARKER_OBSERVATIONS = 30
DEFAULT_DETECTION_THRESHOLD = 95.0
DEFAULT_HEATMAP_RESOLUTION_MM = 10.0

DEFAULT_EXPORT_CSV = True
DEFAULT_EXPORT_JSON = True
DEFAULT_EXPORT_HEATMAP = True


# ==========================================================
# Frame Quality
# ==========================================================

MIN_SHARPNESS = 120.0
MIN_MARKER_SIZE_PX = 30.0
MAX_MARKER_SIZE_PX = 500.0
MAX_VIEW_ANGLE_DEG = 45.0


# ==========================================================
# Calibration Board
# ==========================================================

BOARD_TYPE = "chessboard"  # chessboard | charuco

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_M = 0.025

CHARUCO_SQUARES_X = 9
CHARUCO_SQUARES_Y = 6
CHARUCO_SQUARE_SIZE_M = 0.025
CHARUCO_MARKER_SIZE_M = 0.018
ARUCO_DICT = cv2.aruco.DICT_4X4_1000


# ==========================================================
# Calibration
# ==========================================================

DEFAULT_SAMPLES_NEEDED = 20
CALIBRATION_FILE = "camera_ext.json"
CAPTURE_DELAY = 2.0
MIN_CENTER_SHIFT = 40.0
FLIP_CODE = -1


# ==========================================================
# Command Line Arguments (override layer)
# ==========================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Camera Calibration / Plane Quality Assessment"
    )

    # -------------------------
    # General
    # -------------------------

    parser.add_argument("--mode",
                        type=str,
                        default=ProgramMode.CALIBRATION.value,
                        choices=[m.value for m in ProgramMode])

    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--flip",
                        action="store_true",
                        default=False)

    # -------------------------
    # Calibration
    # -------------------------

    parser.add_argument("--file", type=str, default=CALIBRATION_FILE)
    parser.add_argument("--board", type=str, default=BOARD_TYPE,
                        choices=["chessboard", "charuco"])

    parser.add_argument("--board-width", type=int, default=CHESSBOARD_SIZE[0])
    parser.add_argument("--board-height", type=int, default=CHESSBOARD_SIZE[1])
    parser.add_argument("--square-size", type=float, default=SQUARE_SIZE_M)
    parser.add_argument("--marker-size", type=float, default=CHARUCO_MARKER_SIZE_M)
    #parser.add_argument("--dict", type=str, default=CHARUCO_DICT_NAME)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_NEEDED)

    # -------------------------
    # Plane Quality
    # -------------------------

    parser.add_argument("--plane-frames", type=int, default=DEFAULT_PLANE_FRAMES)
    parser.add_argument("--min-observations", type=int, default=DEFAULT_MIN_MARKER_OBSERVATIONS)
    parser.add_argument("--detection-threshold", type=float, default=DEFAULT_DETECTION_THRESHOLD)
    parser.add_argument("--heatmap-resolution", type=float, default=DEFAULT_HEATMAP_RESOLUTION_MM)

    parser.add_argument("--export-csv", action="store_true", default=DEFAULT_EXPORT_CSV)
    parser.add_argument("--export-json", action="store_true", default=DEFAULT_EXPORT_JSON)
    parser.add_argument("--export-heatmap", action="store_true", default=DEFAULT_EXPORT_HEATMAP)

    # -------------------------
    # Frame Quality
    # -------------------------

    parser.add_argument("--min-sharpness", type=float, default=MIN_SHARPNESS)
    parser.add_argument("--min-marker-size", type=float, default=MIN_MARKER_SIZE_PX)
    parser.add_argument("--max-marker-size", type=float, default=MAX_MARKER_SIZE_PX)
    parser.add_argument("--max-view-angle", type=float, default=MAX_VIEW_ANGLE_DEG)

    return parser.parse_args()