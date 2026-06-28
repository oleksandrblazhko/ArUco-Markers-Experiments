"""
plane_statistics.py

Statistical data structures used by Plane Quality Assessment.

These classes contain aggregated statistics calculated from
multiple PlaneSample objects.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


# ==========================================================
# Statistics for one board position
# ==========================================================

@dataclass(slots=True)
class PlanePointStatistics:
    """
    Aggregated statistics for one position on the calibration plane.
    """

    # Position on board
    board_x_mm: float
    board_y_mm: float

    # Number of observations
    observations: int = 0

    # Number of successful detections
    detections: int = 0

    # Detection probability
    detection_rate: float = 0.0

    # Marker size
    mean_marker_size_px: float = 0.0
    std_marker_size_px: float = 0.0

    # Image sharpness
    mean_sharpness: float = 0.0
    std_sharpness: float = 0.0

    # Distance from image center
    mean_distance_center_px: float = 0.0

    # Camera geometry
    mean_camera_distance_mm: float = 0.0

    mean_pitch_deg: float = 0.0
    mean_yaw_deg: float = 0.0
    mean_roll_deg: float = 0.0

    # Calibration quality
    mean_reprojection_error_px: float = 0.0


# ==========================================================
# Whole experiment statistics
# ==========================================================

@dataclass(slots=True)
class PlaneStatistics:
    """
    Statistics for the whole Plane Quality experiment.
    """

    total_frames: int = 0

    total_samples: int = 0

    total_detected: int = 0

    board_width_mm: float = 0.0

    board_height_mm: float = 0.0

    camera_distance_mm: float = 0.0

    statistics: Dict[
        Tuple[float, float],
        PlanePointStatistics
    ] = field(default_factory=dict)

    @property
    def detection_rate(self) -> float:
        """
        Global detection rate.
        """

        if self.total_samples == 0:
            return 0.0

        return (
            100.0 *
            self.total_detected /
            self.total_samples
        )

    @property
    def number_of_points(self) -> int:
        """
        Number of board positions.
        """

        return len(self.statistics)

    def clear(self):
        """
        Remove all accumulated statistics.
        """

        self.total_frames = 0
        self.total_samples = 0
        self.total_detected = 0

        self.statistics.clear()

    def add_point(
        self,
        point: PlanePointStatistics
    ):
        """
        Store statistics for one board position.
        """

        key = (
            point.board_x_mm,
            point.board_y_mm
        )

        self.statistics[key] = point

    def get_point(
        self,
        board_x_mm: float,
        board_y_mm: float
    ) -> PlanePointStatistics | None:
        """
        Return statistics for one board position.
        """

        return self.statistics.get(
            (board_x_mm, board_y_mm)
        )
    