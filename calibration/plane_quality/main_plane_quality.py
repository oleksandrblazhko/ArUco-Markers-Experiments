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
# UTILITIES
# =========================================================

def generate_hypothetical_grid(rows, cols, marker_size_mm, separation_mm):
    """
    Generates the real-world coordinates (in mm) of the center of each marker in the grid.
    Returns a numpy array of shape (rows*cols, 1, 2).
    """
    world_points = []
    step = marker_size_mm + separation_mm
    for r in range(rows):
        for c in range(cols):
            world_points.append([c * step, r * step])
    
    return np.array(world_points, dtype=np.float32).reshape(-1, 1, 2)



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

def pixel_to_world(pt, H):
    if H is None:
        return 0.0, 0.0

    p = np.array([[pt[0], pt[1]]], dtype=np.float32)
    w = cv2.perspectiveTransform(np.array([p]), H)

    return float(w[0][0][0]), float(w[0][0][1])


# =========================================================
# GRID RECONSTRUCTION HELPERS
# =========================================================

def save_markdown_grid(stats, args, filename="plane_quality_grid.md"):
    """
    Builds a row-by-column markdown grid representation of detection rates
    based on a complete, structured grid.
    """
    if not stats or not stats.statistics:
        print("No stats to save in Markdown grid")
        return

    stats_map = {(p.board_x_mm, p.board_y_mm): p for p in stats.statistics.values()}
    
    grid = [["-" for _ in range(args.grid_cols)] for _ in range(args.grid_rows)]
    step = args.marker_size + args.marker_separation

    for r in range(args.grid_rows):
        for c in range(args.grid_cols):
            world_x = c * step
            world_y = r * step
            
            point_stats = stats_map.get((world_x, world_y))
            if point_stats:
                grid[r][c] = f"{point_stats.detection_rate:.1f}%"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# ArUco Marker Detection Quality Grid ({args.grid_rows}x{args.grid_cols})\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")
        
        headers = ["Row/Col"] + [f"Col {c}" for c in range(args.grid_cols)]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        
        for r in range(args.grid_rows):
            row_data = [f"Row {r}"] + grid[r]
            f.write("| " + " | ".join(row_data) + " |\n")
            
    print(f"Saved Markdown grid report to {filename}")


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
    homography_is_stable = False
    hypothetical_grid_mm = None

    print("Press ESC to stop. Looking for marker plane...")

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            print("WARNING: frame not received")
            continue

        if args.flip:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = detector.detectMarkers(gray)
        
        current_time = time.time()
        
        detections = []
        if ids is not None:
            for i, corners in enumerate(corners_list):
                if int(ids[i][0]) == args.marker_id:
                    detections.append({
                        'id': int(ids[i][0]),
                        'center': center_point(corners),
                        'corners': corners,
                        'size': marker_size_px(corners)
                    })

        if not homography_is_stable:
            cv2.putText(frame, "SEARCHING FOR PLANE...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if len(detections) > 20: 
                all_centers = np.array([d['center'] for d in detections])
                
                tl_idx = np.argmin(np.sum(all_centers, axis=1)) 
                tr_idx = np.argmax(all_centers[:, 0] - all_centers[:, 1]) 
                bl_idx = np.argmin(all_centers[:, 0] - all_centers[:, 1]) 
                br_idx = np.argmax(np.sum(all_centers, axis=1)) 

                image_corners = np.array([
                    detections[tl_idx]['center'],
                    detections[tr_idx]['center'],
                    detections[br_idx]['center'],
                    detections[bl_idx]['center'],
                ], dtype=np.float32)

                step = args.marker_size + args.marker_separation
                world_corners = np.array([
                    [0, 0],
                    [(args.grid_cols - 1) * step, 0],
                    [(args.grid_cols - 1) * step, (args.grid_rows - 1) * step],
                    [0, (args.grid_rows - 1) * step]
                ], dtype=np.float32)

                H_initial, _ = cv2.findHomography(world_corners, image_corners)

                if H_initial is not None:
                    hypothetical_grid_mm = generate_hypothetical_grid(
                        args.grid_rows, args.grid_cols, args.marker_size, args.marker_separation
                    )
                    projected_grid_px = cv2.perspectiveTransform(hypothetical_grid_mm, H_initial)
                    
                    world_pts = []
                    img_pts = []

                    for i, p_proj in enumerate(projected_grid_px):
                        p_proj_tuple = (p_proj[0][0], p_proj[0][1])
                        
                        min_dist = float('inf')
                        best_det_idx = -1
                        for j, det in enumerate(detections):
                            dist = distance(p_proj_tuple, det['center'])
                            if dist < min_dist:
                                min_dist = dist
                                best_det_idx = j
                        
                        if best_det_idx != -1 and min_dist < detections[best_det_idx]['size']:
                            world_pts.append(hypothetical_grid_mm[i][0])
                            img_pts.append(detections[best_det_idx]['center'])
                    
                    if len(img_pts) > 10:
                        H, _ = cv2.findHomography(np.array(world_pts), np.array(img_pts), cv2.RANSAC, 5.0)
                        homography_is_stable = True
                        print("Homography locked.")

        else: 
            plane_frame = PlaneFrame(frame_id=frame_id, timestamp=current_time)
            
            projected_grid_px = cv2.perspectiveTransform(hypothetical_grid_mm, H)
            
            for i, p_proj in enumerate(projected_grid_px):
                p_proj_tuple = (p_proj[0][0], p_proj[0][1])
                world_x, world_y = hypothetical_grid_mm[i][0]

                min_dist = float('inf')
                best_det = None
                for det in detections:
                    dist = distance(p_proj_tuple, det['center'])
                    if dist < min_dist:
                        min_dist = dist
                        best_det = det
                
                sample_detected = False
                sharpness = 0.0
                marker_px_size = 0.0
                cx, cy = p_proj_tuple
                
                if best_det is not None and min_dist < best_det['size'] * 0.75:
                    sample_detected = True
                    cv2.circle(frame, (int(p_proj_tuple[0]), int(p_proj_tuple[1])), 5, (0, 255, 0), -1)
                    
                    corners = best_det['corners']
                    marker_px_size = best_det['size']
                    cx, cy = best_det['center']
                    
                    x1, y1 = np.min(corners[0], axis=0).astype(int)
                    x2, y2 = np.max(corners[0], axis=0).astype(int)
                    sharpness = compute_sharpness(gray, max(0, x1), max(0, y1), 
                                                  min(gray.shape[1], x2), min(gray.shape[0], y2))
                else:
                    cv2.circle(frame, (int(p_proj_tuple[0]), int(p_proj_tuple[1])), 5, (0, 0, 255), 1)

                sample = PlaneSample(
                    frame_id=frame_id,
                    timestamp=current_time,
                    marker_id=i, 
                    detected=sample_detected,
                    board_x_mm=float(world_x),
                    board_y_mm=float(world_y),
                    image_x_px=float(cx),
                    image_y_px=float(cy),
                    distance_from_center_px=distance((cx, cy), (frame.shape[1]/2, frame.shape[0]/2)),
                    marker_size_px=marker_px_size,
                    sharpness=sharpness
                )
                plane_frame.add_sample(sample)
                
            if len(plane_frame.samples) > 0:
                collector.add_frame(plane_frame)

        cv2.imshow("Plane Quality", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        
        frame_id += 1

    print("\nAnalyzing...")
    if len(collector.get_frames()) == 0:
        print("ERROR: No frames collected")
    else:
        stats = analyzer.analyze(collector)
        if not stats or not stats.statistics:
            print("ERROR: Analyzer returned empty stats")
        else:
            reporter.print_summary(stats)
            reporter.save_csv(stats, "plane_quality.csv")
            reporter.save_json(stats, "plane_quality.json")
            save_markdown_grid(stats, args, "plane_quality_grid.md")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    