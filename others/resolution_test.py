import cv2
import time
import numpy as np

# -------- SETTINGS --------
CAM_INDEX = 0
MARKER_SIZE = 0.035  # 35 mm (ВАЖЛИВО: твоє значення)

# ArUco setup (compatible version)
# cv2.aruco.DICT_ARUCO_ORIGINAL), "ARUCO_ORIGINAL (5x5)"),
# cv2.aruco.DICT_4X4_1000), "4X4_1000")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
params = cv2.aruco.DetectorParameters()


def detect_markers(gray):
    """Compatibility wrapper for old/new OpenCV"""
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return detector.detectMarkers(gray)
    else:
        return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)


def run_test(width, height, seconds=15):
    print(f"\n=== TEST {width}x{height} ===")

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    tvec_history = []
    rvec_history = []
    times = []
    lost_frames = 0

    start = time.time()

    # fake calibration (only for comparison)
    fx = width
    fy = height
    cx = width / 2
    cy = height / 2

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist = np.zeros((5, 1))

    while time.time() - start < seconds:

        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detect_markers(gray)

        if ids is not None:

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                MARKER_SIZE,
                camera_matrix,
                dist
            )

            # draw axes
            for i in range(len(rvecs)):
                cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist,
                    rvecs[i],
                    tvecs[i],
                    0.03
                )

            tvec_history.append(tvecs[0][0])
            rvec_history.append(rvecs[0][0])

        else:
            lost_frames += 1

        # FPS display
        fps = 1.0 / (time.time() - t0 + 1e-6)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"{width}x{height}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("ArUco Test", frame)

        times.append(time.time() - t0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()

    # -------- STATS --------

    def stability(data):
        if len(data) < 2:
            return (np.zeros(3), np.zeros(3))
        data = np.array(data)
        return np.mean(data, axis=0), np.std(data, axis=0)

    fps_avg = len(times) / seconds
    avg_time = np.mean(times)

    t_mean, t_std = stability(tvec_history)
    r_mean, r_std = stability(rvec_history)

    print("\n--- RESULTS ---")
    print("FPS:", round(fps_avg, 2))
    print("Frame time:", round(avg_time * 1000, 2), "ms")
    print("Lost frames:", lost_frames)

    print("\nTvec std:", t_std)
    print("Rvec std:", r_std)

    return {
        "fps": fps_avg,
        "t_std": t_std,
        "r_std": r_std,
        "lost": lost_frames
    }


# -------- MAIN --------

if __name__ == "__main__":

    result_640 = run_test(640, 480)
    result_1280 = run_test(1280, 720)

    print("\n======================")
    print("COMPARISON SUMMARY")
    print("======================")

    print("\n640x480 FPS:", result_640["fps"])
    print("1280x720 FPS:", result_1280["fps"])

    print("\nTvec stability (lower = better)")
    print("640:", result_640["t_std"])
    print("1280:", result_1280["t_std"])

    print("\nRvec stability (lower = better)")
    print("640:", result_640["r_std"])
    print("1280:", result_1280["r_std"])