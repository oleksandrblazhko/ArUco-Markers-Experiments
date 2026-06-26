import numpy as np
import cv2
import cv2.aruco as aruco
from collections import deque

def get_marker_bits(dictionary, marker_id, side=4):
    """
    Generates the image of the ArUco marker and extracts its internal bit matrix (excluding the black border).
    """
    img = aruco.generateImageMarker(dictionary, marker_id, 240)
    cell = img.shape[0] // (side + 2)
    bits = np.zeros((side, side), dtype=int)
    
    for r in range(side):
        for c in range(side):
            y0 = (r + 1) * cell
            y1 = (r + 2) * cell
            x0 = (c + 1) * cell
            x1 = (c + 2) * cell
            roi = img[y0:y1, x0:x1]
            # OpenCV generateImageMarker produces binary values: 0 (black), 255 (white).
            # We average the ROI to determine the bit. 1 for white, 0 for black.
            bits[r, c] = 1 if np.mean(roi) > 127 else 0
            
    return bits

def count_ones(m):
    return int(np.sum(m))

def count_transitions(m):
    t = 0
    # Rows
    for r in range(4):
        for c in range(3):
            if m[r, c] != m[r, c + 1]:
                t += 1
    # Columns
    for c in range(4):
        for r in range(3):
            if m[r, c] != m[r + 1, c]:
                t += 1
    return t

def count_chessboard(m):
    count = 0
    for r in range(3):
        for c in range(3):
            block = m[r:r + 2, c:c + 2]
            if np.array_equal(block, np.array([[0, 1], [1, 0]])):
                count += 1
            elif np.array_equal(block, np.array([[1, 0], [0, 1]])):
                count += 1
    return count

def count_isolated(m):
    isolated = 0
    for r in range(4):
        for c in range(4):
            val = m[r, c]
            neigh = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < 4 and 0 <= cc < 4:
                    neigh.append(m[rr, cc])
            if all(n != val for n in neigh):
                isolated += 1
    return isolated

def largest_component(m):
    visited = np.zeros((4, 4), dtype=bool)
    best = 0
    for r in range(4):
        for c in range(4):
            if visited[r, c]:
                continue
            color = m[r, c]
            q = deque([(r, c)])
            visited[r, c] = True
            size = 0
            while q:
                rr, cc = q.popleft()
                size += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = rr + dr
                    nc = cc + dc
                    if (0 <= nr < 4 and 0 <= nc < 4 and 
                            not visited[nr, nc] and m[nr, nc] == color):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            best = max(best, size)
    return best

def homogeneous_score(m):
    score = 0
    for r in range(4):
        if np.all(m[r, :] == m[r, 0]):
            score += 1
    for c in range(4):
        if np.all(m[:, c] == m[0, c]):
            score += 1
    return score

def calculate_perimeter(m):
    """
    Calculates the perimeter of the white cells.
    Any boundary of a white cell (value 1) touching a black cell (0) or the outer border
    (which is black in ArUco) adds 1 to the perimeter.
    """
    perimeter = 0
    for r in range(4):
        for c in range(4):
            if m[r, c] == 1:
                # Check 4 neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < 4 and 0 <= nc < 4):
                        # Border touches outer black frame (which is 0)
                        perimeter += 1
                    elif m[nr, nc] == 0:
                        perimeter += 1
    return perimeter

def count_edge_white_cells(m):
    """
    Counts white cells (1) on the edge of the 4x4 matrix.
    These white cells touch the outer black border of the ArUco marker.
    """
    edge_cells = 0
    for r in range(4):
        for c in range(4):
            if m[r, c] == 1:
                if r == 0 or r == 3 or c == 0 or c == 3:
                    edge_cells += 1
    return edge_cells

def count_connected_components(m, color=1):
    """
    Returns the number of connected components of the given color (1 for white, 0 for black).
    """
    visited = np.zeros((4, 4), dtype=bool)
    count = 0
    for r in range(4):
        for c in range(4):
            if m[r, c] == color and not visited[r, c]:
                count += 1
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr = rr + dr
                        nc = cc + dc
                        if (0 <= nr < 4 and 0 <= nc < 4 and 
                                not visited[nr, nc] and m[nr, nc] == color):
                            visited[nr, nc] = True
                            q.append((nr, nc))
    return count

def extract_features(m):
    """
    Extracts all features as a dictionary.
    """
    ones = count_ones(m)
    white_comp = count_connected_components(m, color=1)
    black_comp = count_connected_components(m, color=0)
    return {
        "Ones": ones,
        "OnesDev": abs(ones - 8),
        "Transitions": count_transitions(m),
        "Chessboard": count_chessboard(m),
        "Isolated": count_isolated(m),
        "Cmax": largest_component(m),
        "Homogeneous": homogeneous_score(m),
        "Perimeter": calculate_perimeter(m),
        "EdgeWhiteCells": count_edge_white_cells(m),
        "Components": white_comp,
        "BlackComponents": black_comp,
        "TotalComponents": white_comp + black_comp
    }
