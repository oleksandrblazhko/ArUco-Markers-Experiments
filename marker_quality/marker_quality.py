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
MARKER_SIZE_M = 0.012  # Marker size in meters
ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
# ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# Build path to camera calibration file relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.dirname(script_dir) # Go up one level from utils
CAMERA_CALIBRATION_FILE = os.path.join(server_dir, '../', 'camera_ext.json')
OUTPUT_JSON_FILE = os.path.join(script_dir, 'marker_test.json') # Save JSON in the same dir as script
MARKER_ANALYSIS_FILE = os.path.join(script_dir, 'marker_quality_test.md') # Save analysis results to a markdown file

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

# --- Main Functions ---

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
        print("WARNING: Camera calibration file not found. Using default values.")
        fx = fy = 800
        cx, cy = 640 / 2, 480 / 2
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1))
        return camera_matrix, dist_coeffs

def record_marker_data(cap, camera_matrix, dist_coeffs, test_duration_s):
    """
    Detects markers and records their pose data for a fixed duration.
    Returns a dictionary with marker data and the total number of frames processed.
    """
    marker_data = defaultdict(lambda: {'tvecs': [], 'rvecs': []})
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
            rvecs, tvecs, _ = estimate_pose_single_markers(corners, MARKER_SIZE_M, camera_matrix, dist_coeffs)

            for i, marker_id in enumerate(ids.flatten()):
                marker_data[marker_id]['tvecs'].append(tvecs[i].flatten())
                marker_data[marker_id]['rvecs'].append(rvecs[i].flatten())
                
    return marker_data, total_frames

def calculate_stats(marker_data, total_frames):
    """
    Calculates std for tvecs/rvecs (RSS noise) and detection rate for each marker.
    Returns a list of dictionaries.
    """
    results = []
    for marker_id, data in marker_data.items():
        frames_detected = len(data['tvecs'])
        if frames_detected < 2:
            print(f"WARNING: Not enough data for marker {marker_id} (< 2 detections). Skipping.")
            continue

        tvecs_arr = np.array(data['tvecs'])
        rvecs_arr = np.array(data['rvecs'])

        # --- Calculate RSS for Pose Stability ---
        std_tvecs = np.std(tvecs_arr, axis=0)
        std_rvecs = np.std(rvecs_arr, axis=0)
        all_stds = np.concatenate((std_tvecs, std_rvecs))
        rss_noise = np.linalg.norm(all_stds)

        # --- Calculate Detection Rate for Detection Stability ---
        detection_rate = (frames_detected / total_frames) * 100 if total_frames > 0 else 0

        print(f"  - Marker ID: {marker_id}, RSS Noise: {rss_noise:.4f}, Detection Rate: {detection_rate:.2f}%")
        results.append({
            'marker_id': marker_id, 
            'rss': rss_noise,
            'detection_rate': detection_rate,
            'std_tvec_x': std_tvecs[0], 'std_tvec_y': std_tvecs[1], 'std_tvec_z': std_tvecs[2],
            'std_rvec_x': std_rvecs[0], 'std_rvec_y': std_rvecs[1], 'std_rvec_z': std_rvecs[2]
        })

    return results

def get_user_input(prompt):
    """Get user input from the console."""
    return input(prompt)

def main():
    """Main function to run the marker quality test."""
    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(description="ArUco Marker Quality Test Script")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height")
    parser.add_argument("--num-groups", type=int, default=16, help="Number of marker groups to test")
    parser.add_argument("--duration", type=int, default=10, help="Duration of the test for each location in seconds")
    args = parser.parse_args()


    camera_matrix, dist_coeffs = load_camera_calibration()

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {CAM_INDEX}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    locations = {
        "center": "у центрі перед камерою",
        "top": "зверху від камери",
        "bottom": "знизу від камери із разворотом на 180 градусів",
        "left": "зліва від камери",
        "right": "зправа від камери із разворотом на 90 градусів вправо"
    }
    angles = [0, 90, 180, 270]
    all_results = []
    
    quit_test = False
    
    for group_num in range(1, args.num_groups + 1):
        if quit_test: break
        print("        " + "="*50)
        print(f"Підготуйте групу маркерів #{group_num} (9 маркерів).")
        
        skip_group = False
        for loc_key, loc_desc in locations.items():
            if quit_test or skip_group: break
            
            for angle in angles:
                if quit_test or skip_group: break
                print("\r\n" + "-"*50)
                print(f"РОЗТАШУВАННЯ: Група #{group_num} - {loc_desc} (кут: {angle}°)")
                play_beep()

                # --- Preview Loop ---
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        quit_test = True
                        break

                    # --- Add Axis Visualization ---
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids = detect_markers(gray)
                    
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                        rvecs, tvecs, _ = estimate_pose_single_markers(corners, MARKER_SIZE_M, camera_matrix, dist_coeffs)
                        for i in range(len(ids)):
                            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.01)
                    
                    cv2.putText(frame, f"Group: {group_num}, Location: {loc_key}, Angle: {angle} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    cv2.putText(frame, f"Resolution: {args.width}x{args.height}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(frame, "Press ENTER to START RECORDING", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(frame, "Press 's' to SKIP to next group", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
                    cv2.putText(frame, "Press 'q' to QUIT", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.imshow('Marker Quality Test Preview', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 13: # Enter
                        break 
                    if key == ord('q'):
                        quit_test = True
                        break
                    if key == ord('s'):
                        # This will break the inner loops and continue to the next group
                        skip_group = True 
                        break
                
                if quit_test or skip_group: break

                # --- Recording and Calculation ---
                marker_data, total_frames = record_marker_data(cap, camera_matrix, dist_coeffs, args.duration)

                if not marker_data:
                    print("No markers detected during recording.")
                    continue

                print(f"\r\nRESULTS for Group {group_num}, Location '{loc_key}', Angle {angle}°:")
                location_stats = calculate_stats(marker_data, total_frames)

                for stat in location_stats:
                    all_results.append({
                        'group': group_num,
                        'location': loc_key,
                        'angle': angle,
                        'marker_id': int(stat['marker_id']),
                        'rss': float(stat['rss']),
                        'detection_rate': float(stat['detection_rate'])
                    })

    # --- Save results to JSON ---
    if all_results:
        print(f"\r\nSaving results to {OUTPUT_JSON_FILE}...")
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as jsonfile:
            json.dump(all_results, jsonfile, indent=4)
        print("Done.")

    # --- Marker Quality Rankings ---
    if all_results:
        print(f"\r\nGenerating analysis report to {MARKER_ANALYSIS_FILE}...")
        
        with open(MARKER_ANALYSIS_FILE, 'w', encoding='utf-8') as md_file:
            def print_and_write(text=""):
                print(text)
                md_file.write(text + "\n")

            print_and_write("# ArUco Marker Quality Analysis Report")
            print_and_write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print_and_write(f"Marker Size: {MARKER_SIZE_M} meters")
            print_and_write(f"Test Duration per Setup: {args.duration} seconds")
            print_and_write("\r\n---")

            # --- 1. Pose Stability Ranking (RSS) ---
            print_and_write("\r\n## 1. Pose Stability Ranking (Lower Average RSS is Better)")
            print_and_write("```")
            
            pose_stats = defaultdict(lambda: {'total_rss': 0.0, 'count': 0})
            for res in all_results:
                pose_stats[res['marker_id']]['total_rss'] += res['rss']
                pose_stats[res['marker_id']]['count'] += 1
            
            pose_ranking = [{'marker_id': mid, 'average_rss': stats['total_rss'] / stats['count']} for mid, stats in pose_stats.items() if stats['count'] > 0]
            pose_ranking_sorted = sorted(pose_ranking, key=lambda x: x['average_rss'])
            
            for rank, entry in enumerate(pose_ranking_sorted, 1):
                print_and_write(f"{rank}. Marker ID: {entry['marker_id']:<4}, Average RSS: {entry['average_rss']:.4f}")
            print_and_write("```")
            
            # --- 2. Detection Stability Ranking (Detection Rate) ---
            print_and_write("\r\n## 2. Detection Stability Ranking (Higher Average Rate is Better)")
            print_and_write("```")

            detection_stats = defaultdict(lambda: {'total_rate': 0.0, 'count': 0})
            for res in all_results:
                detection_stats[res['marker_id']]['total_rate'] += res['detection_rate']
                detection_stats[res['marker_id']]['count'] += 1

            detection_ranking = [{'marker_id': mid, 'average_rate': stats['total_rate'] / stats['count']} for mid, stats in detection_stats.items() if stats['count'] > 0]
            detection_ranking_sorted = sorted(detection_ranking, key=lambda x: x['average_rate'], reverse=True)
            
            for rank, entry in enumerate(detection_ranking_sorted, 1):
                print_and_write(f"{rank}. Marker ID: {entry['marker_id']:<4}, Average Detection Rate: {entry['average_rate']:.2f}%")
            print_and_write("```")

            # --- 3. Quality Stability Ranking (Consistency) ---
            print_and_write("\r\n## 3. Quality Stability Ranking (Lower RSS StdDev is Better)")
            print_and_write("```")

            location_based_stats = defaultdict(list)
            for res in all_results:
                location_based_stats[res['marker_id']].append(res['rss'])

            quality_stability_ranking = []
            for marker_id, rss_list in location_based_stats.items():
                if len(rss_list) > 1:
                    rss_std_dev = np.std(rss_list)
                    quality_stability_ranking.append({'marker_id': marker_id, 'rss_std_dev': rss_std_dev})

            quality_stability_sorted = sorted(quality_stability_ranking, key=lambda x: x['rss_std_dev'])
            
            for rank, entry in enumerate(quality_stability_sorted, 1):
                print_and_write(f"{rank}. Marker ID: {entry['marker_id']:<4}, RSS Std. Dev.: {entry['rss_std_dev']:.4f}")
            print_and_write("```")

        print("Done.")

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("\r\nTest finished.")

if __name__ == "__main__":
    main()
