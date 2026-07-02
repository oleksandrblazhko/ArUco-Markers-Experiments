import argparse

import cv2


# ==========================================================
# Calibration Board
# ==========================================================

ARUCO_DICT = cv2.aruco.DICT_4X4_1000


# ==========================================================
# Command Line Arguments (override layer)
# ==========================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Plane Quality Assessment"
    )

    # -------------------------
    # General
    # -------------------------

    parser.add_argument("--cam", type=int, default=0,
                        help="Camera device ID")
    parser.add_argument("--width", type=int, default=640,
                        help="Camera frame width")
    parser.add_argument("--height", type=int, default=480,
                        help="Camera frame height")
    parser.add_argument("--flip", action="store_true", default=False,
                        help="Flip the camera view horizontally")

    # -------------------------
    # Grid Configuration
    # -------------------------
    
    parser.add_argument("--marker-id", type=int, default=691,
                        help="ID of the ArUco marker used in the grid")
    parser.add_argument("--marker-size", type=float, default=8.0,
                        help="Size of the ArUco marker in mm")
    parser.add_argument("--marker-separation", type=float, default=1.0,
                        help="Size of the white stripe (separation) between markers in mm")
    parser.add_argument("--grid-rows", type=int, default=19,
                        help="Number of rows in the marker grid")
    parser.add_argument("--grid-cols", type=int, default=27,
                        help="Number of columns in the marker grid")

    return parser.parse_args()
