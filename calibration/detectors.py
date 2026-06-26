import cv2
from cv2 import aruco
import numpy as np

def find_chessboard_corners(gray_image, chessboard_size):
    """
    Finds chessboard corners in a grayscale image.
    """
    try:
        # New detector
        found, corners = cv2.findChessboardCornersSB(
            gray_image,
            chessboard_size
        )
    except Exception:
        # Old detector
        found, corners = cv2.findChessboardCorners(
            gray_image,
            chessboard_size
        )

    if found:
        # Refine corner positions
        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
        corners_subpix = cv2.cornerSubPix(
            gray_image,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )
        return True, corners_subpix

    return False, None

def find_charuco_corners(gray_image, charuco_board, dictionary):
    """
    Finds ChArUco corners in a grayscale image.
    """
    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(charuco_board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray_image)
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
            return True, charuco_corners, charuco_ids
    else:
        aruco_params = aruco.DetectorParameters()
        marker_corners, marker_ids, _ = aruco.detectMarkers(
            gray_image,
            dictionary,
            parameters=aruco_params
        )

        if marker_ids is not None and len(marker_ids) > 0:
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray_image,
                charuco_board
            )
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                return True, charuco_corners, charuco_ids
            
    return False, None, None
