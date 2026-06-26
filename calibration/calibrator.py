import cv2
import numpy as np
import json

def perform_calibration(objpoints, imgpoints, image_size):
    """
    Performs camera calibration for chessboard.
    """
    try:
        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None
        )
        return rms, mtx, dist, rvecs, tvecs
    except Exception as e:
        print(f"Calibration error: {e}")
        return None, None, None, None, None

def perform_charuco_calibration(charuco_corners, charuco_ids, board, image_size):
    """
    Performs camera calibration for ChArUco board.
    """
    try:
        rms, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charuco_corners,
            charuco_ids,
            board,
            image_size,
            None,
            None
        )
        return rms, mtx, dist, rvecs, tvecs
    except Exception as e:
        print(f"Calibration error: {e}")
        return None, None, None, None, None

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """
    Calculates the mean reprojection error for chessboard.
    """
    total_error = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            mtx,
            dist
        )
        error = cv2.norm(
            imgpoints[i],
            projected,
            cv2.NORM_L2
        ) / len(projected)
        total_error += error
    
    return total_error / len(objpoints)

def save_calibration_data(filename, mtx, dist):
    """
    Saves calibration data to a JSON file.
    """
    data = {
        "mtx": mtx.tolist(),
        "dist": dist.tolist()
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
