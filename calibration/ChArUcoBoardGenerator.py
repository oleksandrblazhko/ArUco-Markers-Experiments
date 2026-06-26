import cv2
from cv2 import aruco

# =====================================================
# ChArUco Board Generator
# =====================================================

# ---------- Board geometry ----------

SQUARES_X = 9          # cells horizontally
SQUARES_Y = 6           # cells vertically

SQUARE_SIZE_MM = 25.0   # chess square size
MARKER_SIZE_MM = 18.0   # ArUco size

# ---------- Printing ----------

DPI = 300

OUTPUT_FILE = "charuco_board_10x7_A4_300dpi.png"

# ---------- Dictionary ----------

dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_50
)

# ---------- Convert mm -> pixels ----------

MM_TO_INCH = 1.0 / 25.4

square_px = int(
    round(SQUARE_SIZE_MM * MM_TO_INCH * DPI)
)

marker_px = int(
    round(MARKER_SIZE_MM * MM_TO_INCH * DPI)
)

# ---------- Create board ----------

board = aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_SIZE_MM / 1000.0,   # meters
    MARKER_SIZE_MM / 1000.0,   # meters
    dictionary
)

# ---------- Image size ----------

board_width_px = SQUARES_X * square_px
board_height_px = SQUARES_Y * square_px

margin_px = 120

image_width = board_width_px + 2 * margin_px
image_height = board_height_px + 2 * margin_px

# ---------- Render ----------

board_image = board.generateImage(
    (image_width, image_height),
    marginSize=margin_px,
    borderBits=1
)

# ---------- Save ----------

cv2.imwrite(
    OUTPUT_FILE,
    board_image
)

# ---------- Report ----------

board_width_mm = SQUARES_X * SQUARE_SIZE_MM
board_height_mm = SQUARES_Y * SQUARE_SIZE_MM

num_charuco_corners = (
    (SQUARES_X - 1)
    * (SQUARES_Y - 1)
)

print()
print("=== ChArUco Board Generated ===")
print()

print(f"File: {OUTPUT_FILE}")

print()
print("Board geometry:")

print(
    f"  Cells          : "
    f"{SQUARES_X} x {SQUARES_Y}"
)

print(
    f"  Square size    : "
    f"{SQUARE_SIZE_MM:.1f} mm"
)

print(
    f"  Marker size    : "
    f"{MARKER_SIZE_MM:.1f} mm"
)

print()

print(
    f"Physical size    : "
    f"{board_width_mm:.0f} mm x "
    f"{board_height_mm:.0f} mm"
)

print()

print(
    f"ChArUco corners  : "
    f"{num_charuco_corners}"
)

print(
    f"Dictionary       : "
    f"DICT_4X4_50"
)

print()

print(
    f"Image size       : "
    f"{image_width} x {image_height} px"
)

print(
    f"Print resolution : "
    f"{DPI} DPI"
)

print()