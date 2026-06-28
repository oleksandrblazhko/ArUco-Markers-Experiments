"""
plane_quality_analyzer.py

Converts collected PlaneFrame data into statistical model.

Input:
    PlaneQualityCollector

Output:
    PlaneStatistics
"""

import numpy as np

import config

from plane_statistics import (
    PlaneStatistics,
    PlanePointStatistics
)

from plane_quality_collector import PlaneQualityCollector


class PlaneQualityAnalyzer:
    """
    Performs statistical analysis of plane quality dataset.
    """
    def __init__(self):
        self.config = config


    def analyze(
        self,
        collector: PlaneQualityCollector
    ) -> PlaneStatistics:

        stats = PlaneStatistics()

        frames = collector.get_frames()

        stats.total_frames = len(frames)
        stats.total_samples = collector.number_of_samples()

        stats.total_detected = sum(
            1 for s in collector.get_samples()
            if s.detected
        )

        # -----------------------------------------------------
        # GROUP BY BOARD POINT
        # -----------------------------------------------------

        grouped = {}

        for sample in collector.get_samples():

            key = (
                sample.board_x_mm,
                sample.board_y_mm
            )

            if key not in grouped:
                grouped[key] = {
                    "detected": [],
                    "sharpness": [],
                    "size": [],
                    "distance": [],
                    "camera_distance": [],
                    "pitch": [],
                    "yaw": [],
                    "roll": []
                }

            g = grouped[key]

            if sample.detected:
                g["detected"].append(1)
            else:
                g["detected"].append(0)

            g["sharpness"].append(sample.sharpness)
            g["size"].append(sample.marker_size_px)
            g["distance"].append(sample.distance_from_center_px)

            g["camera_distance"].append(sample.camera_distance_mm)
            g["pitch"].append(sample.pitch_deg)
            g["yaw"].append(sample.yaw_deg)
            g["roll"].append(sample.roll_deg)

        # -----------------------------------------------------
        # BUILD STATISTICS PER POINT
        # -----------------------------------------------------

        for (x, y), g in grouped.items():

            point = PlanePointStatistics(
                board_x_mm=x,
                board_y_mm=y,

                observations=len(g["detected"]),
                detections=sum(g["detected"]),

                detection_rate=(
                    100.0 * sum(g["detected"]) / len(g["detected"])
                    if g["detected"] else 0.0
                ),

                mean_marker_size_px=float(np.mean(g["size"])) if g["size"] else 0.0,
                std_marker_size_px=float(np.std(g["size"])) if g["size"] else 0.0,

                mean_sharpness=float(np.mean(g["sharpness"])) if g["sharpness"] else 0.0,
                std_sharpness=float(np.std(g["sharpness"])) if g["sharpness"] else 0.0,

                mean_distance_center_px=float(np.mean(g["distance"])) if g["distance"] else 0.0,

                mean_camera_distance_mm=float(np.mean(g["camera_distance"])) if g["camera_distance"] else 0.0,

                mean_pitch_deg=float(np.mean(g["pitch"])) if g["pitch"] else 0.0,
                mean_yaw_deg=float(np.mean(g["yaw"])) if g["yaw"] else 0.0,
                mean_roll_deg=float(np.mean(g["roll"])) if g["roll"] else 0.0
            )

            stats.add_point(point)

        # -----------------------------------------------------
        # GLOBAL METADATA
        # -----------------------------------------------------

        stats.board_width_mm = self._estimate_board_width(grouped)
        stats.board_height_mm = self._estimate_board_height(grouped)

        return stats

    # =========================================================
    # BOARD GEOMETRY ESTIMATION
    # =========================================================

    def _estimate_board_width(self, grouped) -> float:

        xs = [x for x, _ in grouped.keys()]

        if not xs:
            return 0.0

        return float(max(xs) - min(xs))

    def _estimate_board_height(self, grouped) -> float:

        ys = [y for _, y in grouped.keys()]

        if not ys:
            return 0.0

        return float(max(ys) - min(ys))
    