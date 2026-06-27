import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
import json
from collections import deque

# ==========================================================
# CONFIGURATION
# ==========================================================

JSON_FILE = "marker_detection_rate.json"
OUTPUT_CSV = "aruco_analysis.csv"

DICT = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_250
)

MARKER_SIZE = 4

# ==========================================================
# UTILITIES
# ==========================================================

def get_marker_bits(dictionary, marker_id):

    side = MARKER_SIZE

    img = aruco.generateImageMarker(
        dictionary,
        marker_id,
        200
    )

    # Видаляємо чорну рамку
    cell = img.shape[0] // (side + 2)

    bits = np.zeros((side, side), dtype=np.uint8)

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
# TRANSITIONS
# ==========================================================

def row_transitions(bits):

    count = 0

    rows, cols = bits.shape

    for r in range(rows):
        for c in range(cols - 1):
            if bits[r, c] != bits[r, c + 1]:
                count += 1

    return count


def col_transitions(bits):

    count = 0

    rows, cols = bits.shape

    for c in range(cols):
        for r in range(rows - 1):
            if bits[r, c] != bits[r + 1, c]:
                count += 1

    return count


# ==========================================================
# CONNECTED COMPONENTS
# ==========================================================

def count_components(binary_matrix, target_value):

    rows, cols = binary_matrix.shape

    visited = np.zeros_like(binary_matrix, dtype=bool)

    components = 0

    directions = [
        (-1,0),
        (1,0),
        (0,-1),
        (0,1)
    ]

    for r in range(rows):
        for c in range(cols):

            if visited[r,c]:
                continue

            if binary_matrix[r,c] != target_value:
                continue

            components += 1

            q = deque()
            q.append((r,c))

            visited[r,c] = True

            while q:

                cr, cc = q.popleft()

                for dr, dc in directions:

                    nr = cr + dr
                    nc = cc + dc

                    if (
                        0 <= nr < rows and
                        0 <= nc < cols and
                        not visited[nr,nc] and
                        binary_matrix[nr,nc] == target_value
                    ):
                        visited[nr,nc] = True
                        q.append((nr,nc))

    return components


# ==========================================================
# SYMMETRY
# ==========================================================

def horizontal_symmetry(bits):

    flipped = np.fliplr(bits)

    equal = np.sum(bits == flipped)

    return equal / bits.size


def vertical_symmetry(bits):

    flipped = np.flipud(bits)

    equal = np.sum(bits == flipped)

    return equal / bits.size


# ==========================================================
# ROTATIONAL SIMILARITY
# ==========================================================

def rotation_similarity(bits, k):

    rotated = np.rot90(bits, k)

    equal = np.sum(bits == rotated)

    return equal / bits.size


# ==========================================================
# ANALYSIS
# ==========================================================

def analyze_marker(marker_id, detection_rate):

    bits = get_marker_bits(DICT, marker_id)

    black_bits = int(np.sum(bits == 0))
    white_bits = int(np.sum(bits == 1))

    result = {

        "Marker_ID": marker_id,

        "Detection_Rate":
            detection_rate,

        "BlackBits":
            black_bits,

        "WhiteBits":
            white_bits,

        "RowTransitions":
            row_transitions(bits),

        "ColTransitions":
            col_transitions(bits),

        "BlackComponents":
            count_components(bits, 0),

        "WhiteComponents":
            count_components(bits, 1),

        "HorizontalSymmetry":
            horizontal_symmetry(bits),

        "VerticalSymmetry":
            vertical_symmetry(bits),

        "Rotation90Similarity":
            rotation_similarity(bits, 1),

        "Rotation180Similarity":
            rotation_similarity(bits, 2),

        "Rotation270Similarity":
            rotation_similarity(bits, 3)

    }

    return result


# ==========================================================
# MAIN
# ==========================================================

with open(JSON_FILE, "r", encoding="utf-8") as f:
    markers = json.load(f)

results = []

for item in markers:

    marker_id = item["marker_ID"]

    detection_rate = item["Detection_Rate"]

    row = analyze_marker(
        marker_id,
        detection_rate
    )

    results.append(row)

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(df)

print()
print("Saved:", OUTPUT_CSV)