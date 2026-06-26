import argparse

# ---------- Configuration ----------
BOARD_TYPE = "chessboard"  # "chessboard" or "charuco"

# For chessboard
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_M = 0.025 # meters

# For ChArUco board
CHARUCO_SQUARES_X = 9
CHARUCO_SQUARES_Y = 6
CHARUCO_SQUARE_SIZE_M = 0.025  # meters
CHARUCO_MARKER_SIZE_M = 0.018 # meters
CHARUCO_DICT_NAME = "DICT_4X4_50"


DEFAULT_SAMPLES_NEEDED = 20
CALIBRATION_FILE = "camera_ext.json"

CAPTURE_DELAY = 2.0
MIN_CENTER_SHIFT = 40.0  # pixels
FLIP_CODE = -1

def get_args():
    parser = argparse.ArgumentParser(
        description="Camera Calibration using Chessboard or ChArUco board"
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
        "--file",
        type=str,
        default=CALIBRATION_FILE,
        help="Output calibration file"
    )
    
    parser.add_argument(
        "--board",
        type=str,
        default=BOARD_TYPE,
        help="Type of calibration board: 'chessboard' or 'charuco'"
    )

    parser.add_argument(
        "--board-width",
        type=int,
        default=CHESSBOARD_SIZE[0],
        help="Number of squares / corners horizontally"
    )

    parser.add_argument(
        "--board-height",
        type=int,
        default=CHESSBOARD_SIZE[1],
        help="Number of squares / corners vertically"
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=SQUARE_SIZE_M,
        help="Size of square in meters"
    )

    parser.add_argument(
        "--marker-size",
        type=float,
        default=CHARUCO_MARKER_SIZE_M,
        help="Size of ChArUco markers in meters"
    )

    parser.add_argument(
        "--dict",
        type=str,
        default=CHARUCO_DICT_NAME,
        help="ArUco dictionary name for ChArUco board"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_NEEDED,
        help="Number of samples needed for calibration"
    )

    parser.add_argument(
        "--flip",
        action="store_true",
        default=False,
        help="Flip the camera frame"
    )

    return parser.parse_args()
