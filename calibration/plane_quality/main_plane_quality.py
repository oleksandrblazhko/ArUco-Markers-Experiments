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
from config import get_args, ARUCO_DICT


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
    return float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))


def distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


# =========================================================
# SAFE HOMOGRAPHY INPUT
# =========================================================

def extract_correspondences(corners_list, ids):

    if ids is None or len(corners_list) == 0:
        return None, None

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
# GRID RECONSTRUCTION HELPERS
# =========================================================

def cluster_1d(values, max_clusters, tolerance=15.0):
    """
    Groups a list of 1D coordinates (X or Y pixel values) into clusters
    representing columns or rows. Agglomeratively merges closest coordinates.
    """
    if not values:
        return {}
    sorted_vals = sorted(list(set(values)))
    
    # Initial grouping based on pixel distance tolerance
    groups = []
    for val in sorted_vals:
        if not groups:
            groups.append([val])
        else:
            if val - np.mean(groups[-1]) < tolerance:
                groups[-1].append(val)
            else:
                groups.append([val])
                
    # Merge closest groups until we reach <= max_clusters
    while len(groups) > max_clusters:
        min_diff = float('inf')
        merge_idx = -1
        for i in range(len(groups) - 1):
            diff = np.mean(groups[i+1]) - np.mean(groups[i])
            if diff < min_diff:
                min_diff = diff
                merge_idx = i
        if merge_idx != -1:
            groups[merge_idx].extend(groups[merge_idx+1])
            groups.pop(merge_idx+1)
        else:
            break
            
    val_to_idx = {}
    for idx, grp in enumerate(groups):
        for val in grp:
            val_to_idx[val] = idx
            
    return val_to_idx


def save_markdown_grid(stats, filename="plane_quality_grid.md"):
    """
    Builds a row-by-column markdown grid representation of detection rates.
    """
    points = list(stats.statistics.values())
    if not points:
        print("No points to save in Markdown grid")
        return
        
    xs = [p.board_x_mm for p in points]
    ys = [p.board_y_mm for p in points]
    
    # 19x27 layout grid clustering
    x_to_col = cluster_1d(xs, max_clusters=27, tolerance=15.0)
    y_to_row = cluster_1d(ys, max_clusters=19, tolerance=15.0)
    
    num_cols = max(x_to_col.values()) + 1 if x_to_col else 0
    num_rows = max(y_to_row.values()) + 1 if y_to_row else 0
    
    grid = [["-" for _ in range(num_cols)] for _ in range(num_rows)]
    
    for p in points:
        row = y_to_row[p.board_y_mm]
        col = x_to_col[p.board_x_mm]
        grid[row][col] = f"{p.detection_rate:.1f}%"
        
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# ArUco Marker Detection Quality Grid (19x27)\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")
        f.write(f"Detected Grid Size: {num_rows} Rows x {num_cols} Columns\n\n")
        
        headers = ["Row / Col"] + [f"Col {c}" for c in range(num_cols)]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        
        for r in range(num_rows):
            row_data = [f"Row {r}"] + grid[r]
            f.write("| " + " | ".join(row_data) + " |\n")
            
    print(f"Saved Markdown grid report to {filename}")


# =========================================================
# SPATIAL TRACKER FOR INDENTICAL ID MARKERS
# =========================================================

class TrackedMarker:
    _next_id = 1
    
    def __init__(self, marker_id, center, size, timestamp, window_s=2.0):
        self.id = TrackedMarker._next_id
        TrackedMarker._next_id += 1
        self.marker_id = marker_id
        self.start_center = center
        self.last_center = center
        self.last_size = size
        self.window_s = window_s
        self.history = [(timestamp, True)]
        self.missed_count = 0

    def update(self, detected, timestamp, center=None, size=None):
        if detected:
            self.history.append((timestamp, True))
            self.last_center = center
            if size is not None:
                self.last_size = size
            self.missed_count = 0
        else:
            self.history.append((timestamp, False))
            self.missed_count += 1
        
        # Keep only the last window_s seconds
        cutoff = timestamp - self.window_s
        self.history = [obs for obs in self.history if obs[0] >= cutoff]

    @property
    def detection_rate(self):
        if not self.history:
            return 0.0
        detections = sum(1 for obs in self.history if obs[1])
        return detections / len(self.history)


# =========================================================
# MAIN
# =========================================================

def main():

    args = get_args()

    camera = Camera(
        cam_id=args.cam,
        width=args.width,
        height=args.height
    )

    # 🔴 REAL CAMERA CHECK
    if not camera:
        print("ERROR: Camera init failed")
        return

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(dictionary)

    collector = PlaneQualityCollector()
    analyzer = PlaneQualityAnalyzer()
    reporter = PlaneQualityReport()
    heatmap = PlaneQualityHeatmap()

    frame_id = 0
    H = None

    # Track spatial marker instances dynamically
    tracked_markers = []

    print("Press ESC to stop")

    # =====================================================
    # CAMERA LOOP
    # =====================================================

    while True:

        ok, frame = camera.read()
        if not ok or frame is None:
            print("WARNING: frame not received")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_list, ids, _ = detector.detectMarkers(gray)

        # Get list of current detections
        current_time = time.time()
        detections = []
        if ids is not None and len(corners_list) > 0:
            for j, corners in enumerate(corners_list):
                m_id = int(ids[j][0])
                cx, cy = center_point(corners)
                sz = marker_size_px(corners)
                detections.append({
                    'id': m_id,
                    'center': (cx, cy),
                    'size': sz,
                    'corners': corners,
                    'matched': False
                })

        # Match existing trackers to current detections
        for tracker in tracked_markers:
            best_det_idx = -1
            min_dist = float('inf')
            
            # Spatial proximity threshold: 1.0x of marker size to prevent matching neighbors in dense grids
            threshold = tracker.last_size * 1.0
            
            for idx, det in enumerate(detections):
                if det['matched']:
                    continue
                if det['id'] != tracker.marker_id:
                    continue
                
                # Fast bounding box check to optimize performance for large grids (like 19x27)
                dx = tracker.last_center[0] - det['center'][0]
                if abs(dx) > threshold:
                    continue
                dy = tracker.last_center[1] - det['center'][1]
                if abs(dy) > threshold:
                    continue
                    
                dist = (dx*dx + dy*dy)**0.5
                if dist < threshold and dist < min_dist:
                    min_dist = dist
                    best_det_idx = idx
                    
            if best_det_idx != -1:
                detections[best_det_idx]['matched'] = True
                tracker.update(
                    detected=True,
                    timestamp=current_time,
                    center=detections[best_det_idx]['center'],
                    size=detections[best_det_idx]['size']
                )
                detections[best_det_idx]['tracker'] = tracker
            else:
                tracker.update(detected=False, timestamp=current_time)

        # Create new trackers for unmatched detections
        for det in detections:
            if not det['matched']:
                new_tracker = TrackedMarker(det['id'], det['center'], det['size'], current_time, window_s=2.0)
                tracked_markers.append(new_tracker)
                det['tracker'] = new_tracker

        # =================================================
        # 🔥 VISUALIZATION FIX (IMPORTANT)
        # Custom border colored based on 2s detection frequency
        # =================================================
        for det in detections:
            tracker = det['tracker']
            rate = tracker.detection_rate
            
            # Map rate (0.0 to 1.0) to color:
            # 0.0 (Red: BGR 0,0,255) -> 0.5 (Yellow: BGR 0,255,255) -> 1.0 (Dark Green: BGR 0,128,0)
            if rate < 0.5:
                t = rate / 0.5
                color = (0, int(255 * t), 255)
            else:
                t = (rate - 0.5) / 0.5
                color = (0, int(255 * (1.0 - t) + 128 * t), int(255 * (1.0 - t)))

            # Draw custom border around the corners of the marker
            c = det['corners'][0].astype(np.int32)
            cv2.polylines(frame, [c.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)

        # =================================================
        # 🔥 DRAW UNDETECTED TRACKERS IN RED (2s window)
        # =================================================
        for tracker in tracked_markers:
            was_detected_now = tracker.history[-1][1] if tracker.history else False
            if not was_detected_now:
                cx, cy = tracker.last_center
                sz = tracker.last_size
                half_sz = sz / 2.0
                c_proj = np.array([
                    [cx - half_sz, cy - half_sz],
                    [cx + half_sz, cy - half_sz],
                    [cx + half_sz, cy + half_sz],
                    [cx - half_sz, cy + half_sz]
                ], dtype=np.int32)
                
                # Draw thin red border and cross for missed tracker
                cv2.polylines(frame, [c_proj.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=1)
                cv2.drawMarker(frame, (int(cx), int(cy)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

        plane_frame = PlaneFrame(
            frame_id=frame_id,
            timestamp=current_time
        )

        # =================================================
        # DETECTION VALIDATION & METRIC LOGGING
        # =================================================

        if tracked_markers:
            for tracker in tracked_markers:
                was_detected_now = tracker.history[-1][1] if tracker.history else False
                
                if was_detected_now:
                    # Find the detection that matched this tracker
                    matched_det = None
                    for det in detections:
                        if det.get('tracker') == tracker:
                            matched_det = det
                            break
                    
                    if matched_det is not None:
                        corners = matched_det['corners']
                        cx, cy = matched_det['center']
                        size_px = matched_det['size']
                        
                        x1 = max(0, int(np.min(corners[0][:, 0])))
                        y1 = max(0, int(np.min(corners[0][:, 1])))
                        x2 = min(frame.shape[1] - 1, int(np.max(corners[0][:, 0])))
                        y2 = min(frame.shape[0] - 1, int(np.max(corners[0][:, 1])))
                        sharp = compute_sharpness(gray, x1, y1, x2, y2)
                        
                        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                        dist_center = distance((cx, cy), frame_center)
                        
                        sample = PlaneSample(
                            frame_id=frame_id,
                            timestamp=current_time,
                            marker_id=tracker.id,  # Use unique tracker ID
                            detected=True,
                            board_x_mm=float(tracker.start_center[0]),  # Stable grouping coordinate
                            board_y_mm=float(tracker.start_center[1]),
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
                    # Log undetected sample for this active tracker
                    sample = PlaneSample(
                        frame_id=frame_id,
                        timestamp=current_time,
                        marker_id=tracker.id,
                        detected=False,
                        board_x_mm=float(tracker.start_center[0]),
                        board_y_mm=float(tracker.start_center[1]),
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
        else:
            # If no trackers are active yet, log a dummy sample
            sample = PlaneSample(
                frame_id=frame_id,
                timestamp=current_time,
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

        # Remove trackers that haven't been detected at all in the last 2 seconds
        tracked_markers = [t for t in tracked_markers if any(obs[1] for obs in t.history)]

        # =================================================
        # COLLECT
        # =================================================

        if len(plane_frame.samples) > 0:
            collector.add_frame(plane_frame)

        frame_id += 1

        # =================================================
        # GUI
        # =================================================

        cv2.imshow("Plane Quality", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # =====================================================
    # ANALYSIS
    # =====================================================

    print("\nAnalyzing...")

    if len(collector.get_frames()) == 0:
        print("ERROR: No frames collected")
        camera.release()
        cv2.destroyAllWindows()
        return

    stats = analyzer.analyze(collector)

    if not stats:
        print("ERROR: Analyzer returned empty stats")
        camera.release()
        cv2.destroyAllWindows()
        return

    reporter.print_summary(stats)
    reporter.save_csv(stats, "plane_quality.csv")
    reporter.save_json(stats, "plane_quality.json")
    save_markdown_grid(stats, "plane_quality_grid.md")

    # Plotting and PNG heatmaps are disabled as requested
    # heatmap.plot(stats, mode="detection")
    # heatmap.plot(stats, mode="sharpness")
    # heatmap.plot(stats, mode="size")
    # heatmap.save_png(stats, "heatmap_detection.png", "detection")
    # heatmap.save_png(stats, "heatmap_sharpness.png", "sharpness")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    