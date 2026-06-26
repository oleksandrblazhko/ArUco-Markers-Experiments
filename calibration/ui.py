import cv2

def draw_text(frame, text, position, color=(0, 255, 0), font_scale=1, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )

def draw_corners(frame, chessboard_size, corners, found):
    cv2.drawChessboardCorners(
        frame,
        chessboard_size,
        corners,
        found
    )

def draw_grid(frame, grid_size, visited_cells):
    height, width, _ = frame.shape
    cell_width = width // grid_size[0]
    cell_height = height // grid_size[1]

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x1 = i * cell_width
            y1 = j * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            
            if (i, j) in visited_cells:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)


def show_frame(frame, window_name="Calibration"):
    cv2.imshow(window_name, frame)

def destroy_windows():
    cv2.destroyAllWindows()

def wait_key(delay=1):
    return cv2.waitKey(delay)
