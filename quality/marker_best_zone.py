import cv2
import time
import numpy as np
import os
import json
import argparse
from collections import defaultdict

try:
    import winsound
    def play_beep():
        winsound.Beep(700, 500)
except ImportError:
    def play_beep():
        print("\a", end="", flush=True)

# --- Constants ---
CAM_INDEX = 0
MARKER_SIZE_M = 0.012  # Will be overridden by args.marker_size in main()
ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
WINDOW_NAME = 'ArUco Marker Best Zone Test'

# Build paths
script_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.dirname(script_dir)  # Go up one level from quality
CAMERA_CALIBRATION_FILE = os.path.join(server_dir, 'camera_ext.json')
OUTPUT_JSON_FILE = os.path.join(script_dir, 'marker_zone.json')
MARKER_ZONE_MD_FILE = os.path.join(script_dir, 'marker_zone.md')

# --- OpenCV compatibility wrapper ---
if hasattr(cv2.aruco, "ArucoDetector"):
    ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICTIONARY, ARUCO_PARAMS)
    def detect_markers(gray):
        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
        return corners, ids
else:
    def detect_markers(gray):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICTIONARY, parameters=ARUCO_PARAMS)
        return corners, ids

def estimate_pose_single_markers(corners, marker_size, camera_matrix, dist_coeffs):
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        return cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
    else:
        # Fallback for newer OpenCV versions using solvePnP
        half_size = marker_size / 2.0
        obj_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        rvecs = []
        tvecs = []
        for c in corners:
            _, rvec, tvec = cv2.solvePnP(obj_points, c.reshape((4, 2)), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            rvecs.append(rvec.reshape((1, 3)))
            tvecs.append(tvec.reshape((1, 3)))
        return np.array(rvecs), np.array(tvecs), None

# --- Calibration Loading ---
def load_camera_calibration():    
    """Loads camera calibration data from a JSON file."""
    if os.path.exists(CAMERA_CALIBRATION_FILE):
        with open(CAMERA_CALIBRATION_FILE, 'r') as f:
            data = json.load(f)
            camera_matrix = np.array(data["mtx"])
            dist_coeffs = np.array(data["dist"])
            print("INFO: Camera calibration loaded from a file.")
            return camera_matrix, dist_coeffs
    else:
        print(CAMERA_CALIBRATION_FILE)
        print("WARNING: Camera calibration file not found. Using default values.")
        fx = fy = 800
        cx, cy = 640 / 2, 480 / 2
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1))
        return camera_matrix, dist_coeffs

# --- Group Registration ---
def register_group_markers(cap, camera_matrix, dist_coeffs, group_num, marker_size_mm):
    """
    Runs a preview loop to let the user place all 9 markers of the group.
    Once the user is satisfied, they press Enter to lock in the IDs.
    Constructs a grid map of the 3x3 layout.
    Returns:
      - expected_ids: list of marker IDs
      - grid_map: dict mapping ID to (row, col)
    """
    print(f"\n==========================================")
    print(f"РЕЄСТРАЦІЯ ГРУПИ МАРКЕРІВ #{group_num}")
    print(f"Помістіть групу з 9 маркерів перед камерою.")
    print(f"Натисніть ENTER для підтвердження списку маркерів.")
    print(f"Натисніть 'c' для очищення списку виявлених.")
    print(f"Натисніть 'q' для виходу.")
    print(f"==========================================\n")
    
    detected_ids_set = set()
    quit_test = False
    
    # We will keep the last detected corners and ids to build the grid
    last_corners = None
    last_ids = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame from camera.")
            return None, None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers(gray)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = estimate_pose_single_markers(corners, marker_size_mm / 1000.0, camera_matrix, dist_coeffs)
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.01)
                
            for marker_id in ids.flatten():
                detected_ids_set.add(int(marker_id))
                
            last_corners = corners
            last_ids = ids
                
        # Draw registration info on the frame
        cv2.putText(frame, f"Registering Group #{group_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Detected markers: {len(detected_ids_set)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display the list of registered IDs
        ids_str = ", ".join(map(str, sorted(list(detected_ids_set))))
        if len(ids_str) > 50:
            ids_str = ids_str[:47] + "..."
        cv2.putText(frame, f"IDs: {ids_str}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, "Press ENTER to CONFIRM", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to CLEAR detected list", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        cv2.putText(frame, "Press 'q' to QUIT", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13: # Enter
            if len(detected_ids_set) != 9:
                print(f"УВАГА: Виявлено {len(detected_ids_set)} маркерів замість 9.")
            break
        elif key == ord('c'):
            detected_ids_set.clear()
            last_corners = None
            last_ids = None
            print("Список виявлених маркерів очищено.")
        elif key == ord('q'):
            quit_test = True
            break
            
    if quit_test or not detected_ids_set:
        return None, None
        
    expected_ids = sorted(list(detected_ids_set))
    grid_map = {}
    
    # Try to build grid map from the last frame where exactly 9 markers were detected
    built_grid = False
    if last_ids is not None and len(last_ids) == 9:
        try:
            marker_centers = []
            for i in range(len(last_ids)):
                mid = int(last_ids[i][0])
                c = last_corners[i][0]
                mean_pt = np.mean(c, axis=0)
                marker_centers.append({'id': mid, 'center': mean_pt})
                
            # Sort by Y coordinate
            marker_centers.sort(key=lambda m: m['center'][1])
            # Split into 3 rows
            row0 = sorted(marker_centers[0:3], key=lambda m: m['center'][0])
            row1 = sorted(marker_centers[3:6], key=lambda m: m['center'][0])
            row2 = sorted(marker_centers[6:9], key=lambda m: m['center'][0])
            
            for col, m in enumerate(row0): grid_map[m['id']] = (0, col)
            for col, m in enumerate(row1): grid_map[m['id']] = (1, col)
            for col, m in enumerate(row2): grid_map[m['id']] = (2, col)
            built_grid = True
        except Exception as e:
            print(f"Помилка побудови сітки 3x3: {e}. Використовується автоматичне сортування.")
            
    if not built_grid:
        # Fallback grid map: assign row-major order based on sorted IDs
        for idx, mid in enumerate(expected_ids[:9]):
            row = idx // 3
            col = idx % 3
            grid_map[mid] = (row, col)
            
    return expected_ids, grid_map

# --- Estimate Group Center ---
def estimate_group_center(corners, ids, grid_map, marker_size_mm, border_size_mm, camera_matrix, dist_coeffs):
    """
    Estimates the 3D group center and projects it to 2D pixel coordinates.
    """
    if ids is None or grid_map is None:
        return None, None
        
    unit_size_m = (marker_size_mm + 2 * border_size_mm) / 1000.0
    rvecs, tvecs, _ = estimate_pose_single_markers(corners, marker_size_mm / 1000.0, camera_matrix, dist_coeffs)
    
    centers_3d = []
    for i, marker_id in enumerate(ids.flatten()):
        marker_id = int(marker_id)
        if marker_id in grid_map:
            row, col = grid_map[marker_id]
            # Offset of marker relative to center marker (1, 1)
            offset_x = (col - 1) * unit_size_m
            offset_y = (row - 1) * unit_size_m
            
            rvec = rvecs[i].flatten()
            tvec = tvecs[i].flatten()
            
            R, _ = cv2.Rodrigues(rvec)
            # Center of the board in camera coords
            center_3d = tvec - R.dot(np.array([offset_x, offset_y, 0.0]))
            centers_3d.append(center_3d)
            
    if not centers_3d:
        return None, None
        
    avg_center_3d = np.mean(centers_3d, axis=0)
    
    # Project 3D center to 2D
    img_pts, _ = cv2.projectPoints(avg_center_3d.reshape(1, 1, 3), np.zeros((3,1)), np.zeros((3,1)), camera_matrix, dist_coeffs)
    center_2d = img_pts.reshape(2)
    return avg_center_3d, (int(center_2d[0]), int(center_2d[1]))

# --- Data Recording and Stats ---
def record_marker_data(cap, camera_matrix, dist_coeffs, test_duration_s, grid_map, marker_size_mm, border_size_mm):
    """
    Detects markers, records pose data, and tracks corner points and group centers.
    """
    marker_data = defaultdict(lambda: {'tvecs': [], 'rvecs': []})
    corners_points = []
    center_points = []
    total_frames = 0
    start_time = time.time()

    print(f"Recording data for {test_duration_s} seconds...")

    while time.time() - start_time < test_duration_s:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame from camera.")
            break
        
        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers(gray)

        if ids is not None:
            rvecs, tvecs, _ = estimate_pose_single_markers(corners, marker_size_mm / 1000.0, camera_matrix, dist_coeffs)

            for i, marker_id in enumerate(ids.flatten()):
                marker_data[marker_id]['tvecs'].append(tvecs[i].flatten())
                marker_data[marker_id]['rvecs'].append(rvecs[i].flatten())
                for pt in corners[i].reshape(-1, 2):
                    corners_points.append((float(pt[0]), float(pt[1])))
                    
            # Estimate 3D and 2D group center
            _, center_2d = estimate_group_center(corners, ids, grid_map, marker_size_mm, border_size_mm, camera_matrix, dist_coeffs)
            if center_2d is not None:
                center_points.append(center_2d)
                
    return marker_data, total_frames, corners_points, center_points

def calculate_detection_rates(marker_data, total_frames, expected_ids):
    """
    Calculates detection rate for each expected marker ID.
    Returns:
      - individual_rates: dict of {marker_id: detection_rate}
      - group_average: float (average detection rate of all expected markers)
    """
    individual_rates = {}
    for marker_id in expected_ids:
        frames_detected = len(marker_data[marker_id]['tvecs']) if marker_id in marker_data else 0
        detection_rate = (frames_detected / total_frames) * 100 if total_frames > 0 else 0.0
        individual_rates[marker_id] = detection_rate
    
    if expected_ids:
        group_average = sum(individual_rates.values()) / len(expected_ids)
    else:
        group_average = 0.0
        
    return individual_rates, group_average

# --- Draw Completed Locations ---
def draw_completed_locations(frame, completed_locations):
    """Draws blue quadrilaterals representing already measured marker groups."""
    for loc in completed_locations:
        pts = np.array(loc['corners_2d'], dtype=np.int32)
        # Draw blue quadrilateral
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
        
        # Draw location ID and rate at the center
        cx, cy = loc['center_2d']
        cv2.putText(frame, f"L{loc['location_id']}", (int(cx) - 15, int(cy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(frame, f"{loc['avg_detection_rate']:.1f}%", (int(cx) - 15, int(cy) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# --- Best Zone Calculation ---
def calculate_best_zone(completed_locations):
    """Calculates axis-aligned pixel bounding box for locations within 1.0% of maximum DR."""
    if not completed_locations:
        return None
        
    # Find max rate
    max_rate = max(loc['avg_detection_rate'] for loc in completed_locations)
    
    # Best locations: within 1.0% of max_rate
    best_locs = [loc for loc in completed_locations if loc['avg_detection_rate'] >= max_rate - 1.0]
    
    if not best_locs:
        return None
        
    # Collect all 2D corners of best locations
    all_pts = []
    for loc in best_locs:
        all_pts.extend(loc['corners_2d'])
        
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    
    return (x_min, y_min, x_max, y_max)

# --- Markdown Report Writer ---
def write_markdown_report(completed_locations, args):
    """Generates a structured markdown report from completed locations."""
    total_group_size_mm = 3 * args.marker_size + 6 * args.border_size
    with open(MARKER_ZONE_MD_FILE, 'w', encoding='utf-8') as f:
        f.write("# Звіт про якість детектування маркерів по зонах (ArUco Marker Zone Quality Report)\n\n")
        f.write(f"**Дата створення:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Параметри конфігурації:**\n")
        f.write(f"- Відстань камери від поверхні: {args.cam_distance} мм\n")
        f.write(f"- Тривалість запису: {args.duration} сек\n")
        f.write(f"- Роздільна здатність камери: {args.width}x{args.height}\n")
        f.write(f"- Розмір маркера: {args.marker_size} мм (біла дужка: {args.border_size} мм)\n")
        f.write(f"- Розмір групи маркерів (3х3): {total_group_size_mm}x{total_group_size_mm} мм\n\n")
        f.write("---\n\n")
        
        f.write("## 1. Результати вимірювань по локаціях (Measured Locations)\n\n")
        f.write("| Локація | Середній DR (%) | Відстань Z (мм) | Деталі по кутах (Кут: DR%) |\n")
        f.write("| :---: | :---: | :---: | :--- |\n")
        
        for loc in completed_locations:
            avg_z_mm = loc['avg_z_m'] * 1000.0
            
            # Format angle details
            angle_details = []
            for angle, rate in sorted(loc['angle_rates'].items()):
                angle_details.append(f"{angle}°: {rate:.1f}%")
            details_str = ", ".join(angle_details)
            
            f.write(f"| #{loc['location_id']} | {loc['avg_detection_rate']:.2f}% | {avg_z_mm:.1f} | {details_str} |\n")
            
        f.write("\n---\n\n")
        
        # Best Zone
        best_box = calculate_best_zone(completed_locations)
        if best_box is not None:
            f.write("## 2. Аналіз та найкраща зона детектування (Best Detection Zone)\n\n")
            max_rate = max(loc['avg_detection_rate'] for loc in completed_locations)
            best_loc_ids = [str(loc['location_id']) for loc in completed_locations if loc['avg_detection_rate'] >= max_rate - 1.0]
            
            f.write(f"- **Максимальний Detection Rate:** {max_rate:.2f}%\n")
            f.write(f"- **Локації найкращої зони (DR >= {max_rate - 1.0:.2f}%):** #{', #'.join(best_loc_ids)}\n")
            f.write(f"- **Координати найкращої зони на екрані (пікселі):** X: [{best_box[0]} .. {best_box[2]}], Y: [{best_box[1]} .. {best_box[3]}]\n\n")

def save_results(completed_locations, args):
    """Saves raw json and generates/updates the markdown report."""
    serialized_locs = []
    for loc in completed_locations:
        serialized_locs.append({
            'location_id': loc['location_id'],
            'corners_2d': [[float(p[0]), float(p[1])] for p in loc['corners_2d']],
            'center_2d': [float(loc['center_2d'][0]), float(loc['center_2d'][1])],
            'avg_detection_rate': float(loc['avg_detection_rate']),
            'avg_z_m': float(loc['avg_z_m']),
            'angle_rates': {str(k): float(v) for k, v in loc['angle_rates'].items()}
        })
        
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as jsonfile:
        json.dump(serialized_locs, jsonfile, indent=4)
        
    write_markdown_report(completed_locations, args)

# --- Console Analysis and Output Summary ---
def analyze_and_print_summary(completed_locations):
    """Prints recommendations about the best zone and results to console."""
    if not completed_locations:
        return
        
    print("\n" + "="*50)
    print("АНАЛІЗ РЕЗУЛЬТАТІВ ТЕСТУВАННЯ / TEST RESULTS ANALYSIS")
    print("="*50)
    
    max_rate = max(loc['avg_detection_rate'] for loc in completed_locations)
    best_locs = [loc for loc in completed_locations if loc['avg_detection_rate'] >= max_rate - 1.0]
    
    print(f"\nМаксимальний середній Detection Rate: {max_rate:.2f}%")
    print("Локації найкращої зони:")
    for loc in best_locs:
        avg_z_mm = loc['avg_z_m'] * 1000.0
        print(f"  * Локація #{loc['location_id']}: {loc['avg_detection_rate']:.2f}% (Середня відстань: {avg_z_mm:.1f} мм)")
        
    best_box = calculate_best_zone(completed_locations)
    if best_box is not None:
        print(f"\nКоординати найкращої зони на екрані (пікселі):")
        print(f"  X: [{best_box[0]} .. {best_box[2]}]")
        print(f"  Y: [{best_box[1]} .. {best_box[3]}]")
        
    print("\n" + "="*50 + "\n")

# --- Results Preview Visualization ---
def run_results_preview(cap, camera_matrix, dist_coeffs, completed_locations, grid_map, args):
    """
    Keeps the program running after the test and displays the live feed
    with blue quadrilaterals for all measured locations and a green rectangle
    representing the best zone.
    """
    if not completed_locations:
        print("No completed locations to display.")
        return
        
    print("\n" + "="*50)
    print("ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ / RESULTS VISUALIZATION")
    print("Програма показує виміряні локації (синій) та найкращу зону (зелений).")
    print("Натисніть 'q' для виходу з програми.")
    print("="*50 + "\n")
    
    best_box = calculate_best_zone(completed_locations)
    total_group_size_mm = 3 * args.marker_size + 6 * args.border_size
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw detected markers live
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = estimate_pose_single_markers(corners, args.marker_size / 1000.0, camera_matrix, dist_coeffs)
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.01)
                
            # Draw live estimated group center
            _, live_ctr = estimate_group_center(corners, ids, grid_map, args.marker_size, args.border_size, camera_matrix, dist_coeffs)
            if live_ctr is not None:
                cv2.drawMarker(frame, live_ctr, (255, 0, 255), cv2.MARKER_CROSS, 15, 2)
                
        # Draw all completed locations (Blue quadrilaterals)
        draw_completed_locations(frame, completed_locations)
        
        # Draw Best Zone (Green)
        if best_box is not None:
            cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 3)
            cv2.putText(frame, "BEST ZONE (NAIKRASHCHA ZONA)", (best_box[0], max(15, best_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Display instructions
        cv2.putText(frame, "VISUALIZATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Cam Distance: {args.cam_distance}mm | Group size: {total_group_size_mm}x{total_group_size_mm}mm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "Blue: Measured Locations | Green: Best Zone", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to QUIT program", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# --- Main Program ---
def main():
    global MARKER_SIZE_M
    parser = argparse.ArgumentParser(description="ArUco Marker Best Zone Interactive Detection Script")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height")
    parser.add_argument("--num-groups", type=int, default=1, help="Number of marker groups to test")
    parser.add_argument("--duration", type=int, default=5, help="Duration of the test for each setup in seconds")
    parser.add_argument("--marker-size", type=float, default=12.0, help="Marker size in mm (size of the black square)")
    parser.add_argument("--border-size", type=float, default=1.0, help="White border size around the marker in mm")
    parser.add_argument("--cam-distance", type=float, default=450.0, help="Camera distance from the surface center in mm")
    args = parser.parse_args()

    MARKER_SIZE_M = args.marker_size / 1000.0
    total_group_size_mm = 3 * args.marker_size + 6 * args.border_size

    camera_matrix, dist_coeffs = load_camera_calibration()

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {CAM_INDEX}")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize single window at startup
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    completed_locations = []
    location_id = 1
    quit_experiment = False

    for group_num in range(1, args.num_groups + 1):
        if quit_experiment:
            break
            
        # 1. Register the expected marker IDs in the group and construct grid layout
        expected_ids, grid_map = register_group_markers(cap, camera_matrix, dist_coeffs, group_num, args.marker_size)
        if expected_ids is None or len(expected_ids) == 0:
            print(f"Registration skipped or cancelled for Group #{group_num}.")
            continue
            
        print(f"Зареєстровано {len(expected_ids)} маркерів для групи #{group_num}: {expected_ids}")
        
        while not quit_experiment:
            print("\n" + "="*50)
            print(f"НАЛАШТУВАННЯ ЛОКАЦІЇ #{location_id}")
            print("Розмістіть групу маркерів у новій частині зони.")
            print("Натисніть ENTER для початку вимірювань для цієї локації.")
            print("Натисніть 'q' для припинення експерименту та перегляду аналізу.")
            print("="*50 + "\n")
            
            # Location setup preview loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    quit_experiment = True
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids = detect_markers(gray)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    rvecs, tvecs, _ = estimate_pose_single_markers(corners, args.marker_size / 1000.0, camera_matrix, dist_coeffs)
                    for i in range(len(ids)):
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.01)
                    
                    # Show live center
                    _, live_ctr = estimate_group_center(corners, ids, grid_map, args.marker_size, args.border_size, camera_matrix, dist_coeffs)
                    if live_ctr is not None:
                        cv2.drawMarker(frame, live_ctr, (255, 0, 255), cv2.MARKER_CROSS, 15, 2)
                        
                # Draw already completed locations
                draw_completed_locations(frame, completed_locations)
                
                # Draw UI text
                cv2.putText(frame, f"Setup Location #{location_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, "Press ENTER to START 4-ANGLE TEST", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, "Press 'q' to FINISH Experiment", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 13: # Enter
                    break
                elif key == ord('q'):
                    quit_experiment = True
                    break
                    
            if quit_experiment:
                break
                
            # We start the 4-angle loop for the current location
            location_results = []
            angle_rates = {}
            first_rvec = None
            first_tvec = None
            
            angles = [0, 90, 180, 270]
            skip_location = False
            
            for angle in angles:
                if quit_experiment or skip_location:
                    break
                    
                print("\n" + "-"*50)
                print(f"ЛОКАЦІЯ #{location_id} | Кут {angle}°")
                play_beep()
                
                # Preview loop for this angle
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        quit_experiment = True
                        break
                        
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids = detect_markers(gray)
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                        rvecs, tvecs, _ = estimate_pose_single_markers(corners, args.marker_size / 1000.0, camera_matrix, dist_coeffs)
                        for i in range(len(ids)):
                            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.01)
                        
                        # Show live center
                        _, live_ctr = estimate_group_center(corners, ids, grid_map, args.marker_size, args.border_size, camera_matrix, dist_coeffs)
                        if live_ctr is not None:
                            cv2.drawMarker(frame, live_ctr, (255, 0, 255), cv2.MARKER_CROSS, 15, 2)
                            
                    # Draw already completed locations
                    draw_completed_locations(frame, completed_locations)
                    
                    # UI text
                    cv2.putText(frame, f"Location #{location_id} | Angle: {angle} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(frame, "Press ENTER to START RECORDING", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(frame, "Press 's' to SKIP this location", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                    cv2.putText(frame, "Press 'q' to FINISH Experiment", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 13: # Enter
                        break
                    elif key == ord('s'):
                        skip_location = True
                        break
                    elif key == ord('q'):
                        quit_experiment = True
                        break
                        
                if quit_experiment or skip_location:
                    break
                    
                # Recording
                marker_data, total_frames, corners_pts, center_pts = record_marker_data(
                    cap, camera_matrix, dist_coeffs, args.duration,
                    grid_map, args.marker_size, args.border_size
                )
                
                # Calculate stats
                individual_rates, group_avg = calculate_detection_rates(marker_data, total_frames, expected_ids)
                
                print(f"Результати для Кута {angle}°:")
                print(f"  Середній DR групи: {group_avg:.2f}%")
                
                angle_rates[angle] = group_avg
                
                # Average tvec/rvec for center tracking
                if marker_data:
                    all_tvecs = []
                    all_rvecs = []
                    for mid, data in marker_data.items():
                        if data['tvecs']:
                            all_tvecs.extend(data['tvecs'])
                        if data['rvecs']:
                            all_rvecs.extend(data['rvecs'])
                    if all_tvecs:
                        avg_tvec = np.mean(all_tvecs, axis=0)
                        avg_rvec = np.mean(all_rvecs, axis=0)
                        
                        location_results.append({
                            'angle': angle,
                            'group_average': group_avg,
                            'tvec': avg_tvec,
                            'rvec': avg_rvec
                        })
                        
                        if first_tvec is None:
                            first_tvec = avg_tvec
                            first_rvec = avg_rvec

            if skip_location:
                print(f"Локацію #{location_id} пропущено.")
                continue
                
            if quit_experiment:
                break
                
            # Calculate overall location stats if we got measurements
            if location_results:
                avg_rate = sum(r['group_average'] for r in location_results) / len(location_results)
                all_zs = [r['tvec'][2] for r in location_results if r['tvec'] is not None]
                avg_z_m = sum(all_zs) / len(all_zs) if all_zs else args.cam_distance / 1000.0
                
                # Project physical board corners from the first measured pose
                half_w = total_group_size_mm / 2000.0
                local_corners = np.array([
                    [-half_w, -half_w, 0.0],
                    [ half_w, -half_w, 0.0],
                    [ half_w,  half_w, 0.0],
                    [-half_w,  half_w, 0.0]
                ], dtype=np.float32)
                
                img_pts, _ = cv2.projectPoints(
                    local_corners.reshape(4, 1, 3), 
                    first_rvec.reshape(3, 1), 
                    first_tvec.reshape(3, 1), 
                    camera_matrix, 
                    dist_coeffs
                )
                loc_corners_2d = img_pts.reshape(4, 2)
                loc_center_2d = np.mean(loc_corners_2d, axis=0)
                
                completed_locations.append({
                    'location_id': location_id,
                    'corners_2d': loc_corners_2d,
                    'center_2d': loc_center_2d,
                    'avg_detection_rate': avg_rate,
                    'avg_z_m': avg_z_m,
                    'angle_rates': angle_rates
                })
                
                # Save incrementally
                save_results(completed_locations, args)
                print(f"Локація #{location_id} успішно виміряна. Середній DR: {avg_rate:.2f}%")
                location_id += 1

    # --- Dynamic analysis summary output ---
    analyze_and_print_summary(completed_locations)
    
    # --- Results preview visualization ---
    if completed_locations:
        run_results_preview(cap, camera_matrix, dist_coeffs, completed_locations, grid_map, args)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest finished.")

if __name__ == "__main__":
    main()
