"""
plane_sample.py

Data structures used by Plane Quality Assessment.

The module is intentionally independent of OpenCV.
It only stores measurement results collected from each frame.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PlaneSample:
    """
    Single observation of one marker on the calibration plane.

    One object corresponds to one detected (or expected) marker
    in one video frame.
    """

    # ---------------------------------------------------------
    # Frame information
    # ---------------------------------------------------------

    frame_id: int

    timestamp: float

    # ---------------------------------------------------------
    # Marker information
    # ---------------------------------------------------------

    marker_id: int

    detected: bool

    # ---------------------------------------------------------
    # Marker position on calibration board
    # ---------------------------------------------------------

    board_x_mm: float

    board_y_mm: float

    # ---------------------------------------------------------
    # Marker center in image
    # ---------------------------------------------------------

    image_x_px: float

    image_y_px: float

    # ---------------------------------------------------------
    # Distance from image center
    # ---------------------------------------------------------

    distance_from_center_px: float

    # ---------------------------------------------------------
    # Marker geometry
    # ---------------------------------------------------------

    marker_size_px: float

    # ---------------------------------------------------------
    # Local image quality
    # ---------------------------------------------------------

    sharpness: float

    # ---------------------------------------------------------
    # Camera pose relative to board
    # ---------------------------------------------------------

    camera_distance_mm: float

    pitch_deg: float

    yaw_deg: float

    roll_deg: float

    # ---------------------------------------------------------
    # Optional calibration quality
    # ---------------------------------------------------------

    reprojection_error_px: Optional[float] = None

    # ---------------------------------------------------------
    # Optional notes
    # ---------------------------------------------------------

    comment: str = ""

    # ---------------------------------------------------------

    @property
    def board_position(self) -> tuple[float, float]:
        """Return board coordinates."""

        return (self.board_x_mm, self.board_y_mm)

    @property
    def image_position(self) -> tuple[float, float]:
        """Return image coordinates."""

        return (self.image_x_px, self.image_y_px)

    @property
    def was_detected(self) -> bool:
        """Readable alias."""

        return self.detected
    