"""
plane_frame.py

Represents a single captured frame in Plane Quality experiment.
"""

from dataclasses import dataclass, field
from typing import List

from plane_sample import PlaneSample


@dataclass(slots=True)
class PlaneFrame:
    """
    One video frame containing multiple marker observations.
    """

    frame_id: int
    timestamp: float

    # Camera pose (optional but very useful)
    camera_distance_mm: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0

    # All marker observations in this frame
    samples: List[PlaneSample] = field(default_factory=list)

    def add_sample(self, sample: PlaneSample) -> None:
        """
        Add marker observation to this frame.
        """

        self.samples.append(sample)

    def detection_count(self) -> int:
        """
        Number of successfully detected markers.
        """

        return sum(1 for s in self.samples if s.detected)

    def detection_rate(self) -> float:
        """
        Detection rate in this frame.
        """

        if not self.samples:
            return 0.0

        return 100.0 * self.detection_count() / len(self.samples)
    