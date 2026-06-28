"""
plane_quality_collector.py

Frame-based collector for Plane Quality Assessment.

Stores PlaneFrame objects instead of raw samples.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from plane_frame import PlaneFrame
from plane_sample import PlaneSample

class PlaneQualityCollector:
    def __init__(self, config):
        self.grid_rows = config.GRID_ROWS
        self.grid_cols = config.GRID_COLS

    """
    Collects PlaneFrame objects and provides access
    to both frame-level and sample-level data.
    """

    # =========================================================
    # RESET
    # =========================================================

    def reset(self) -> None:

        self._frames: List[PlaneFrame] = []

        self._frames_by_id: Dict[int, PlaneFrame] = {}

        # Flattened cache (optional fast access)
        self._samples: List[PlaneSample] = []

        self._samples_by_point: Dict[
            Tuple[float, float],
            List[PlaneSample]
        ] = defaultdict(list)

    # =========================================================
    # FRAME OPERATIONS
    # =========================================================

    def add_frame(self, frame: PlaneFrame) -> None:
        """
        Add a full frame with all samples.
        """

        self._frames.append(frame)
        self._frames_by_id[frame.frame_id] = frame

        for sample in frame.samples:

            self._samples.append(sample)

            key = (
                sample.board_x_mm,
                sample.board_y_mm
            )

            self._samples_by_point[key].append(sample)

    # ---------------------------------------------------------

    def get_frame(self, frame_id: int) -> PlaneFrame | None:
        """
        Get frame by ID.
        """

        return self._frames_by_id.get(frame_id)

    # ---------------------------------------------------------

    def get_frames(self) -> List[PlaneFrame]:

        return self._frames

    # ---------------------------------------------------------

    def get_last_frame(self) -> PlaneFrame | None:

        if not self._frames:
            return None

        return self._frames[-1]

    # =========================================================
    # SAMPLE ACCESS (flattened view)
    # =========================================================

    def get_samples(self) -> List[PlaneSample]:
        """
        All samples across all frames.
        """

        return self._samples

    # ---------------------------------------------------------

    def get_samples_for_point(
        self,
        board_x_mm: float,
        board_y_mm: float
    ) -> List[PlaneSample]:
        """
        Get all samples for a board position.
        """

        return self._samples_by_point.get(
            (board_x_mm, board_y_mm),
            []
        )

    # =========================================================
    # STATISTICS HELPERS
    # =========================================================

    def number_of_frames(self) -> int:

        return len(self._frames)

    # ---------------------------------------------------------

    def number_of_samples(self) -> int:

        return len(self._samples)

    # ---------------------------------------------------------

    def number_of_points(self) -> int:

        return len(self._samples_by_point)

    # ---------------------------------------------------------

    def global_detection_rate(self) -> float:
        """
        Detection rate across all frames/samples.
        """

        if not self._samples:
            return 0.0

        detected = sum(
            1 for s in self._samples
            if s.detected
        )

        return 100.0 * detected / len(self._samples)

    # =========================================================
    # FRAME ANALYTICS
    # =========================================================

    def frame_detection_rates(self) -> Dict[int, float]:
        """
        Detection rate per frame.
        """

        result = {}

        for frame in self._frames:

            result[frame.frame_id] = frame.detection_rate()

        return result

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self) -> dict:
        """
        Experiment summary.
        """

        return {

            "frames": self.number_of_frames(),

            "samples": self.number_of_samples(),

            "points": self.number_of_points(),

            "global_detection_rate": self.global_detection_rate(),

            "frame_detection_rates": self.frame_detection_rates()

        }

    # =========================================================
    # ITERATION
    # =========================================================

    def __len__(self):

        return len(self._frames)

    def __iter__(self):

        return iter(self._frames)