import cv2
import time
import numpy as np

from plane_sample import PlaneSample
from plane_frame import PlaneFrame
from plane_quality_collector import PlaneQualityCollector
from plane_quality_analyzer import PlaneQualityAnalyzer
from plane_quality_report import PlaneQualityReport
from plane_quality_heatmap import PlaneQualityHeatmap

from camera import Camera
from config import get_args


# =========================================================
# REAL WORLD LAYOUT (mm)
# =========================================================

MARKER_LAYOUT_MM = {
    0: (0.0, 0.0),
    1: (50.0, 0.0),
    2: (100.0, 0.0),
    3: (0.0, 50.0),
    4: (50.0, 50.0),
    5: (100.0, 50.0),
}


# =========================================================
# UTILITIES
# =========================================================

def compute_sharpness(gray, x1, y1, x2, y2):
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return cv2.Laplacian(roi, cv2.CV_64F).var()


def marker_size_px(corners):
    c = corners[0]
    return (
        np.linalg.norm(c[0] - c[1]) +
        np.linalg.norm(c[1] - c[2]) +
        np.linalg.norm(c[2] - c[3]) +
        np.linalg.norm(c[3] - c[0])
    ) / 4.0


def center_point(corners):
    c = corners[0]
    return (
        float(np.mean(c[:, 0])),
        float(np.mean(c[:, 1]))
    )


def distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


# =========================================================
# HOMOGRAPHY HELPERS
# =========================================================

def extract_correspondences(corners_list, ids):

    img_pts = []
    world_pts = []

    for i, corners in enumerate(corners_list):

        marker_id = int(ids[i][0])

        if marker_id not in MARKER_LAYOUT_MM:
            continue

        c = corners[0]

        cx = float(np.mean(c[:, 0]))
        cy = float(np.mean(c[:, 1]))

        img_pts.append([cx, cy])
        world_pts.append(MARKER_LAYOUT_MM[marker_id])

    if len(img_pts) < 4:
        return None, None

    return (
        np.array(img_pts, dtype=np.float32),
        np.array(world_pts, dtype=np.float32)
    )


def pixel_to_world(pt, H):

    if H is None:
        return 0.0, 0.0

    p = np.array([[pt[0], pt[1]]], dtype=np.float32)
    w = cv2.perspectiveTransform(np.array([p]), H)

    return float(w[0][0][0]), float(w[0][0][1])


# =========================================================
# MAIN
# =========================================================

def main():

    # ---------------- Args (FROM OLD PROJECT STYLE) ----------------
    args = get_args()

    # ---------------- Camera ----------------
    camera = Camera(
        cam_id=args.cam,
        width=args.width,
        height=args.height
    )

    # ---------------- Detector ----------------
    DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    }

    dictionary = cv2.aruco.getPredefinedDictionary(
        DICT_MAP.get(args.dict, cv2.aruco.DICT_4X4_50)
    )

    detector = cv2.aruco.ArucoDetector(dictionary)

    # ---------------- Pipeline ----------------
    collector = PlaneQualityCollector()
    analyzer = PlaneQualityAnalyzer()
    reporter = PlaneQualityReport()
    heatmap = PlaneQualityHeatmap()

    frame_id = 0
    H = None

    print("Press ESC to stop")

    # =====================================================
    # CAPTURE LOOP
    # =====================================================

    while True:

        ok, frame = camera.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_list, ids, _ = detector.detectMarkers(gray)

        plane_frame = PlaneFrame(
            frame_id=frame_id,
            timestamp=time.time()
        )

        # -------------------------------------------------
        # HOMOGRAPHY UPDATE
        # -------------------------------------------------

        if ids is not None:

            img_pts, world_pts = extract_correspondences(corners_list, ids)

            if img_pts is not None and len(img_pts) >= 4:
                H, _ = cv2.findHomography(img_pts, world_pts, cv2.RANSAC)

            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

            for i, corners in enumerate(corners_list):

                marker_id = int(ids[i][0])
                cx, cy = center_point(corners)

                x1 = int(np.min(corners[0][:, 0]))
                y1 = int(np.min(corners[0][:, 1]))
                x2 = int(np.max(corners[0][:, 0]))
                y2 = int(np.max(corners[0][:, 1]))

                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                sharp = compute_sharpness(gray, x1, y1, x2, y2)
                size_px = marker_size_px(corners)

                frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                dist_center = distance((cx, cy), frame_center)

                world_x, world_y = pixel_to_world((cx, cy), H)

                sample = PlaneSample(
                    frame_id=frame_id,
                    timestamp=time.time(),

                    marker_id=marker_id,
                    detected=True,

                    board_x_mm=world_x,
                    board_y_mm=world_y,

                    image_x_px=float(cx),
                    image_y_px=float(cy),

                    distance_from_center_px=dist_center,
                    marker_size_px=size_px,
                    sharpness=sharp,

                    camera_distance_mm=0.0,
                    pitch_deg=0.0,
                    yaw_deg=0.0,
                    roll_deg=0.0
                )

                plane_frame.add_sample(sample)

        else:

            sample = PlaneSample(
                frame_id=frame_id,
                timestamp=time.time(),
                marker_id=-1,
                detected=False,
                board_x_mm=0.0,
                board_y_mm=0.0,
                image_x_px=0.0,
                image_y_px=0.0,
                distance_from_center_px=0.0,
                marker_size_px=0.0,
                sharpness=0.0,
                camera_distance_mm=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
                roll_deg=0.0
            )

            plane_frame.add_sample(sample)

        collector.add_frame(plane_frame)

        cv2.imshow("Plane Quality (ArUco)", frame)

        if cv2.waitKey(1) == 27:
            break

        frame_id += 1

    # =====================================================
    # ANALYSIS
    # =====================================================

    print("\nAnalyzing...")

    stats = analyzer.analyze(collector)

    reporter.print_summary(stats)
    reporter.save_csv(stats, "plane_quality.csv")
    reporter.save_json(stats, "plane_quality.json")

    heatmap.plot(stats, mode="detection")
    heatmap.plot(stats, mode="sharpness")
    heatmap.plot(stats, mode="size")

    heatmap.save_png(stats, "heatmap_detection.png", "detection")
    heatmap.save_png(stats, "heatmap_sharpness.png", "sharpness")

    # =====================================================

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
