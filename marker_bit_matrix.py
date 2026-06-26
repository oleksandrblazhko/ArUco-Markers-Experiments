import cv2
import cv2.aruco as aruco
import numpy as np
import json

# ==========================================================
# CONFIG
# ==========================================================

JSON_FILE = "marker_detection_rate.json"
OUTPUT_FILE = "marker_bit_matrix.txt"

DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
MARKER_SIZE = 4

# ==========================================================
# BIT MATRIX EXTRACTION
# ==========================================================

def get_marker_bits(dictionary, marker_id):

    img = aruco.generateImageMarker(dictionary, marker_id, 200)

    side = MARKER_SIZE

    # прибираємо зовнішню рамку
    cell = img.shape[0] // (side + 2)

    bits = np.zeros((side, side), dtype=int)

    for r in range(side):
        for c in range(side):

            y0 = (r + 1) * cell
            y1 = (r + 2) * cell

            x0 = (c + 1) * cell
            x1 = (c + 2) * cell

            roi = img[y0:y1, x0:x1]

            mean_val = np.mean(roi)

            bits[r, c] = 1 if mean_val > 127 else 0

    return bits

# ==========================================================
# MAIN
# ==========================================================

def main():

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        markers = json.load(f)

    # 1. sort by detection rate (descending)
    markers_sorted = sorted(
        markers,
        key=lambda x: x["Detection_Rate"],
        reverse=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for item in markers_sorted:

            marker_id = item["marker_ID"]
            detection_rate = item["Detection_Rate"]

            bits = get_marker_bits(DICT, marker_id)

            # header
            out.write(f"Marker ID: {marker_id} | Detection Rate: {detection_rate}\n")

            # matrix
            for row in bits:
                out.write(" ".join(str(int(x)) for x in row) + "\n")

            out.write("-" * 60 + "\n")

    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()