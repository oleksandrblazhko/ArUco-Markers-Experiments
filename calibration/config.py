import argparse
from enum import Enum

# ==========================================================
# Program Mode
# ==========================================================

class ProgramMode(Enum):
    CALIBRATION = "calibration"
    PLANE_QUALITY = "plane_quality"


# ==========================================================
# Calibration Board
# ==========================================================

BOARD_TYPE = "chessboard"      # chessboard | charuco

# Chessboard

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_M = 0.025

# ChArUco

CHARUCO_SQUARES_X = 9
CHARUCO_SQUARES_Y = 6

CHARUCO_SQUARE_SIZE_M = 0.025
CHARUCO_MARKER_SIZE_M = 0.018

CHARUCO_DICT_NAME = "DICT_4X4_50"

# ==========================================================
# Calibration
# ==========================================================

DEFAULT_SAMPLES_NEEDED = 20

CALIBRATION_FILE = "camera_ext.json"

CAPTURE_DELAY = 2.0

MIN_CENTER_SHIFT = 40.0

FLIP_CODE = -1

# ==========================================================
# Plane Quality Assessment
# ==========================================================

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
# Command Line
# ==========================================================

def get_args():

    parser = argparse.ArgumentParser(
        description="Camera Calibration / Plane Quality Assessment"
    )

    # ------------------------------------------------------
    # General
    # ------------------------------------------------------

    parser.add_argument(
        "--mode",
        type=str,
        default=ProgramMode.CALIBRATION.value,
        choices=[m.value for m in ProgramMode],
        help="Program mode"
    )

    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Camera index"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height"
    )

    parser.add_argument(
        "--flip",
        action="store_true",
        default=False,
        help="Flip camera frame"
    )

    # ------------------------------------------------------
    # Calibration
    # ------------------------------------------------------

    parser.add_argument(
        "--file",
        type=str,
        default=CALIBRATION_FILE,
        help="Calibration output file"
    )

    parser.add_argument(
        "--board",
        type=str,
        default=BOARD_TYPE,
        choices=["chessboard", "charuco"],
        help="Calibration board type"
    )

    parser.add_argument(
        "--board-width",
        type=int,
        default=CHESSBOARD_SIZE[0],
        help="Board width"
    )

    parser.add_argument(
        "--board-height",
        type=int,
        default=CHESSBOARD_SIZE[1],
        help="Board height"
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=SQUARE_SIZE_M,
        help="Square size (meters)"
    )

    parser.add_argument(
        "--marker-size",
        type=float,
        default=CHARUCO_MARKER_SIZE_M,
        help="Marker size (meters)"
    )

    parser.add_argument(
        "--dict",
        type=str,
        default=CHARUCO_DICT_NAME,
        help="ArUco dictionary"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_NEEDED,
        help="Calibration samples"
    )

    # ------------------------------------------------------
    # Plane Quality
    # ------------------------------------------------------

    parser.add_argument(
        "--plane-frames",
        type=int,
        default=DEFAULT_PLANE_FRAMES,
        help="Frames to analyze for Plane Quality"
    )

    parser.add_argument(
        "--min-observations",
        type=int,
        default=DEFAULT_MIN_MARKER_OBSERVATIONS,
        help="Minimum observations for each board position"
    )

    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=DEFAULT_DETECTION_THRESHOLD,
        help="Minimum acceptable detection rate (%)"
    )

    parser.add_argument(
        "--heatmap-resolution",
        type=float,
        default=DEFAULT_HEATMAP_RESOLUTION_MM,
        help="Heatmap interpolation resolution (mm)"
    )

    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=DEFAULT_EXPORT_CSV,
        help="Export CSV report"
    )

    parser.add_argument(
        "--export-json",
        action="store_true",
        default=DEFAULT_EXPORT_JSON,
        help="Export JSON report"
    )

    parser.add_argument(
        "--export-heatmap",
        action="store_true",
        default=DEFAULT_EXPORT_HEATMAP,
        help="Export heatmap image"
    )

    # ------------------------------------------------------
    # Frame Quality
    # ------------------------------------------------------

    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=MIN_SHARPNESS,
        help="Minimum acceptable sharpness"
    )

    parser.add_argument(
        "--min-marker-size",
        type=float,
        default=MIN_MARKER_SIZE_PX,
        help="Minimum marker size (pixels)"
    )

    parser.add_argument(
        "--max-marker-size",
        type=float,
        default=MAX_MARKER_SIZE_PX,
        help="Maximum marker size (pixels)"
    )

    parser.add_argument(
        "--max-view-angle",
        type=float,
        default=MAX_VIEW_ANGLE_DEG,
        help="Maximum board viewing angle"
    )

    return parser.parse_args()
