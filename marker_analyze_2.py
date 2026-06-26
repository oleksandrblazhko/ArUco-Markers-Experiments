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
OUTPUT_CSV = "marker_analysyze_2.csv"

DICT = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_250
)

MARKER_SIZE = 4

# ==========================================================
# MARKER BITS
# ==========================================================

def get_marker_bits(dictionary, marker_id):

    img = aruco.generateImageMarker(
        dictionary,
        marker_id,
        200
    )

    side = MARKER_SIZE

    cell = img.shape[0] // (side + 2)

    bits = np.zeros((side, side), dtype=np.uint8)

    for r in range(side):
        for c in range(side):

            y0 = (r + 1) * cell
            y1 = (r + 2) * cell

            x0 = (c + 1) * cell
            x1 = (c + 2) * cell

            roi = img[y0:y1, x0:x1]

            bits[r, c] = (
                1 if np.mean(roi) > 127 else 0
            )

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

def count_components(matrix, target):

    rows, cols = matrix.shape

    visited = np.zeros_like(
        matrix,
        dtype=bool
    )

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

            if matrix[r,c] != target:
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
                        matrix[nr,nc] == target
                    ):
                        visited[nr,nc] = True
                        q.append((nr,nc))

    return components

# ==========================================================
# SYMMETRY
# ==========================================================

def horizontal_symmetry(bits):

    flipped = np.fliplr(bits)

    return np.sum(
        bits == flipped
    ) / bits.size


def vertical_symmetry(bits):

    flipped = np.flipud(bits)

    return np.sum(
        bits == flipped
    ) / bits.size

# ==========================================================
# ROTATIONAL SIMILARITY
# ==========================================================

def rotation_similarity(bits, k):

    rotated = np.rot90(bits, k)

    return np.sum(
        bits == rotated
    ) / bits.size

# ==========================================================
# HAMMING DISTANCE
# ==========================================================

def hamming_distance(bits1, bits2):

    return int(
        np.sum(bits1 != bits2)
    )


def compute_hamming_metrics(
    all_bits,
    marker_id
):

    current = all_bits[marker_id]

    distances = []

    for other_id, other_bits in all_bits.items():

        if other_id == marker_id:
            continue

        distances.append(
            hamming_distance(
                current,
                other_bits
            )
        )

    return (
        min(distances),
        float(np.mean(distances))
    )

# ==========================================================
# LOAD JSON
# ==========================================================

with open(
    JSON_FILE,
    "r",
    encoding="utf-8"
) as f:

    markers = json.load(f)

# ==========================================================
# PRECOMPUTE ALL BIT MATRICES
# ==========================================================

all_bits = {}

for item in markers:

    marker_id = item["marker_ID"]

    all_bits[marker_id] = get_marker_bits(
        DICT,
        marker_id
    )

# ==========================================================
# ANALYSIS
# ==========================================================

results = []

for item in markers:

    marker_id = item["marker_ID"]

    detection_rate = item["Detection_Rate"]

    bits = all_bits[marker_id]

    black_bits = int(
        np.sum(bits == 0)
    )

    white_bits = int(
        np.sum(bits == 1)
    )

    row_tr = row_transitions(bits)
    col_tr = col_transitions(bits)

    black_comp = count_components(
        bits,
        0
    )

    white_comp = count_components(
        bits,
        1
    )

    nearest_hamming, avg_hamming = (
        compute_hamming_metrics(
            all_bits,
            marker_id
        )
    )

    results.append({

        "Marker_ID":
            marker_id,

        "Detection_Rate":
            detection_rate,

        "BlackBits":
            black_bits,

        "WhiteBits":
            white_bits,

        "FillRatio":
            white_bits / bits.size,

        "RowTransitions":
            row_tr,

        "ColTransitions":
            col_tr,

        "TotalTransitions":
            row_tr + col_tr,

        "BlackComponents":
            black_comp,

        "WhiteComponents":
            white_comp,

        "TotalComponents":
            black_comp + white_comp,

        "HorizontalSymmetry":
            horizontal_symmetry(bits),

        "VerticalSymmetry":
            vertical_symmetry(bits),

        "Rotation90Similarity":
            rotation_similarity(bits, 1),

        "Rotation180Similarity":
            rotation_similarity(bits, 2),

        "Rotation270Similarity":
            rotation_similarity(bits, 3),

        "NearestHammingDistance":
            nearest_hamming,

        "AverageHammingDistance":
            avg_hamming
    })

# ==========================================================
# DATAFRAME
# ==========================================================

df = pd.DataFrame(results)

df = df.sort_values(
    by="Detection_Rate",
    ascending=False
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ==========================================================
# OUTPUT
# ==========================================================

pd.set_option(
    "display.max_columns",
    None
)

print(df)

print("\nSaved:", OUTPUT_CSV)

# ==========================================================
# CORRELATION ANALYSIS
# ==========================================================

print("\n")
print("=" * 70)
print("CORRELATION WITH DETECTION RATE")
print("=" * 70)

corr = (
    df.corr(
        numeric_only=True
    )["Detection_Rate"]
      .sort_values(
          ascending=False
      )
)

print(corr)