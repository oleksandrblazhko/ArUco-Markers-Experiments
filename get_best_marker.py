import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd

# ==========================================================
# PARAMETERS
# ==========================================================

DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

START_ID = 0
END_ID = 40

MARKER_BITS = 4

# ==========================================================
# BIT EXTRACTION
# ==========================================================

def get_marker_bits(marker_id):
    """
    Повертає 4x4 матрицю бітів маркера.
    """

    img = aruco.generateImageMarker(
        DICT,
        marker_id,
        60  # 6 клітинок × 10 px
    )

    cell = img.shape[0] // 6

    bits = np.zeros((4, 4), dtype=np.uint8)

    for r in range(4):
        for c in range(4):

            y0 = (r + 1) * cell
            y1 = (r + 2) * cell

            x0 = (c + 1) * cell
            x1 = (c + 2) * cell

            roi = img[y0:y1, x0:x1]

            bits[r, c] = int(np.mean(roi) < 128)

    return bits


# ==========================================================
# METRICS
# ==========================================================

def count_transitions(bits):

    transitions = 0

    for row in bits:
        transitions += np.sum(row[:-1] != row[1:])

    for col in bits.T:
        transitions += np.sum(col[:-1] != col[1:])

    return int(transitions)


def balance_score(bits):

    black = np.sum(bits)

    ratio = black / bits.size

    return 1.0 - abs(ratio - 0.5) / 0.5


def center_mass_score(bits):

    ys, xs = np.where(bits == 1)

    if len(xs) == 0:
        return 0

    cx = np.mean(xs)
    cy = np.mean(ys)

    ideal = (MARKER_BITS - 1) / 2

    dist = np.sqrt(
        (cx - ideal) ** 2 +
        (cy - ideal) ** 2
    )

    max_dist = np.sqrt(2 * ideal ** 2)

    return 1.0 - dist / max_dist


def symmetry_penalty(bits):

    penalty = 0

    if np.array_equal(bits, np.fliplr(bits)):
        penalty += 1

    if np.array_equal(bits, np.flipud(bits)):
        penalty += 1

    if np.array_equal(bits, bits.T):
        penalty += 1

    return penalty


def corner_activity(bits):

    return np.mean([
        bits[0, 0],
        bits[0, -1],
        bits[-1, 0],
        bits[-1, -1]
    ])


def entropy_score(bits):

    p = np.mean(bits)

    if p == 0 or p == 1:
        return 0

    return -(p*np.log2(p) + (1-p)*np.log2(1-p))


# ==========================================================
# ANALYSIS
# ==========================================================

results = []

for marker_id in range(START_ID, END_ID + 1):

    bits = get_marker_bits(marker_id)

    transitions = count_transitions(bits)

    balance = balance_score(bits)

    center = center_mass_score(bits)

    symmetry = symmetry_penalty(bits)

    corners = corner_activity(bits)

    entropy = entropy_score(bits)

    score = (
        transitions * 2.0 +
        balance * 10.0 +
        center * 5.0 +
        corners * 3.0 +
        entropy * 10.0 -
        symmetry * 8.0
    )

    results.append({
        "ID": marker_id,
        "Transitions": transitions,
        "Balance": round(balance, 3),
        "CenterMass": round(center, 3),
        "Corners": round(corners, 3),
        "Entropy": round(entropy, 3),
        "SymmetryPenalty": symmetry,
        "Score": round(score, 3)
    })

df = pd.DataFrame(results)

df = df.sort_values(
    by="Score",
    ascending=False
)

print(df)

df.to_csv(
    "aruco_quality_ranking.csv",
    index=False
)

print("\nSaved to aruco_quality_ranking.csv")