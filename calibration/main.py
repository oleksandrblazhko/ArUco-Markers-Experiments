import cv2
from cv2 import aruco
import numpy as np
from . import config
from .camera import CameraManager
from .detectors import find_chessboard_corners, find_charuco_corners
from .sample_collector import SampleCollector
from .calibrator import (
    perform_calibration,
    perform_charuco_calibration,
    calculate_reprojection_error,
    save_calibration_data
)
from .ui import (
    draw_text,
    draw_corners,
    draw_grid,
    show_frame,
    destroy_windows,
    wait_key,
)
from .quality_analyzer import QualityAnalyzer

try:
    import winsound
    def play_beep():
        winsound.Beep(1000, 200)
except ImportError:
    def play_beep():
        print("\a", end="", flush=True)

def get_charuco_board(args):
    dictionary = aruco.getPredefinedDictionary(
        getattr(aruco, args.dict)
    )
    return aruco.CharucoBoard(
        (args.board_width, args.board_height),
        args.square_size,
        args.marker_size,
        dictionary
    )

def main():
    args = config.get_args()
    
    try:
        cam_manager = CameraManager(args.cam, args.width, args.height)
    except IOError as e:
        print(e)
        return

    board_type = args.board
    if board_type == 'charuco':
        board = get_charuco_board(args)
        collector = SampleCollector()
        status_text = "No ChArUco board"
        print("=== Camera Calibration with ChArUco board ===")
        print(f"Board: {args.board_width}x{args.board_height}")
    else:
        board = None
        collector = SampleCollector((args.board_width, args.board_height), args.square_size)
        status_text = "No chessboard"
        print("=== Camera Calibration with Chessboard ===")
        print(f"Chessboard: {args.board_width}x{args.board_height}")

    analyzer = QualityAnalyzer(args.width, args.height)
    
    print(f"Need {args.samples} samples")
    print("Move board around the image")
    print("Press q to quit")

    image_size = None
    
    while collector.get_sample_count() < args.samples:
        frame = cam_manager.get_frame()
        if frame is None:
            print("Frame capture error")
            break

        if args.flip:
            frame = cv2.flip(frame, config.FLIP_CODE)

        draw_grid(frame, analyzer.grid_size, analyzer.grid_visits)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        if board_type == 'charuco':
            found, corners, charuco_ids = find_charuco_corners(gray, board, board.getDictionary())
            if found:
                draw_corners(frame, (args.board_width, args.board_height), corners, found)
                captured, status_text = collector.add_sample(
                    corners, config.MIN_CENTER_SHIFT, config.CAPTURE_DELAY, charuco_ids=charuco_ids
                )
                if captured:
                    analyzer.add_sample(corners)
                    play_beep()
        else:
            found, corners = find_chessboard_corners(gray, (args.board_width, args.board_height))
            if found:
                draw_corners(frame, (args.board_width, args.board_height), corners, found)
                captured, status_text = collector.add_sample(
                    corners, config.MIN_CENTER_SHIFT, config.CAPTURE_DELAY
                )
                if captured:
                    analyzer.add_sample(corners)
                    play_beep()

        if not found:
            status_text = f"No {board_type}"
        
        draw_text(
            frame,
            f"Samples: {collector.get_sample_count()}/{args.samples}",
            (20, 40)
        )
        draw_text(
            frame,
            status_text,
            (20, 80),
            color=(0, 0, 255),
            font_scale=0.8
        )
        
        tilt_diversity = np.std(analyzer.aspect_ratios) if analyzer.aspect_ratios else 0
        draw_text(
            frame,
            f"Tilt score: {tilt_diversity:.3f}",
            (20, 160),
            color=(0, 255, 255),
            font_scale=0.8
        )

        draw_text(
            frame,
            f"Grid Visited: {len(analyzer.grid_visits)}/{analyzer.grid_size[0] * analyzer.grid_size[1]}",
            (20, 120),
            color=(255, 0, 0),
            font_scale=0.8
        )
        show_frame(frame)

        key = wait_key()
        if key & 0xFF == ord("q"):
            print("Cancelled")
            cam_manager.release()
            destroy_windows()
            return
    
    cam_manager.release()
    destroy_windows()

    if collector.get_sample_count() < 10:
        print("Too few samples")
        return

    if board_type == 'charuco':
        charuco_corners, charuco_ids = collector.get_points()
        print("Performing calibration...")
        rms, mtx, dist, rvecs, tvecs = perform_charuco_calibration(
            charuco_corners, charuco_ids, board, image_size
        )
    else:
        objpoints, imgpoints = collector.get_points()
        print("Performing calibration...")
        rms, mtx, dist, rvecs, tvecs = perform_calibration(
            objpoints, imgpoints, image_size
        )

    if 'mtx' not in locals() or mtx is None:
        return

    # Reprojection error calculation needs to be adapted for charuco
    # For now, we skip it for charuco
    if board_type != 'charuco':
        reprojection_error = calculate_reprojection_error(
            objpoints, imgpoints, rvecs, tvecs, mtx, dist
        )
        analyzer.generate_report(reprojection_error)
    else:
        analyzer.generate_report(rms)

    save_calibration_data(args.file, mtx, dist)
    print(f"\nSaved: {args.file}")
    
    # Validation is also board specific
    # run_validation(args, mtx, dist)


if __name__ == "__main__":
    main()
