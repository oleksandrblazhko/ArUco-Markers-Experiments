"""
plane_quality_report.py

Exports PlaneStatistics into human-readable formats:
CSV, JSON, and text summary.

This module does NOT perform analysis.
It only serializes results.
"""

import csv
import json
from datetime import datetime

from plane_statistics import PlaneStatistics


class PlaneQualityReport:
    """
    Creates reports from PlaneStatistics.
    """

    # =========================================================
    # CSV EXPORT
    # =========================================================

    def save_csv(
        self,
        stats: PlaneStatistics,
        filename: str
    ) -> None:

        with open(filename, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([
                "board_x_mm",
                "board_y_mm",
                "observations",
                "detections",
                "detection_rate",
                "mean_sharpness",
                "std_sharpness",
                "mean_marker_size_px",
                "std_marker_size_px",
                "mean_distance_center_px",
                "mean_camera_distance_mm",
                "mean_pitch_deg",
                "mean_yaw_deg",
                "mean_roll_deg"
            ])

            for point in stats.statistics.values():

                writer.writerow([
                    point.board_x_mm,
                    point.board_y_mm,
                    point.observations,
                    point.detections,
                    point.detection_rate,
                    point.mean_sharpness,
                    point.std_sharpness,
                    point.mean_marker_size_px,
                    point.std_marker_size_px,
                    point.mean_distance_center_px,
                    point.mean_camera_distance_mm,
                    point.mean_pitch_deg,
                    point.mean_yaw_deg,
                    point.mean_roll_deg
                ])

    # =========================================================
    # JSON EXPORT
    # =========================================================

    def save_json(
        self,
        stats: PlaneStatistics,
        filename: str
    ) -> None:

        data = {

            "timestamp": datetime.now().isoformat(),

            "summary": {
                "frames": stats.total_frames,
                "samples": stats.total_samples,
                "detections": stats.total_detected,
                "global_detection_rate": stats.detection_rate,
                "board_points": stats.number_of_points
            },

            "board": {
                "width_mm": stats.board_width_mm,
                "height_mm": stats.board_height_mm
            },

            "points": []
        }

        for point in stats.statistics.values():

            data["points"].append({

                "board_x_mm": point.board_x_mm,
                "board_y_mm": point.board_y_mm,

                "observations": point.observations,
                "detections": point.detections,
                "detection_rate": point.detection_rate,

                "mean_sharpness": point.mean_sharpness,
                "std_sharpness": point.std_sharpness,

                "mean_marker_size_px": point.mean_marker_size_px,
                "std_marker_size_px": point.std_marker_size_px,

                "mean_distance_center_px": point.mean_distance_center_px,

                "mean_camera_distance_mm": point.mean_camera_distance_mm,

                "mean_pitch_deg": point.mean_pitch_deg,
                "mean_yaw_deg": point.mean_yaw_deg,
                "mean_roll_deg": point.mean_roll_deg
            })

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # =========================================================
    # TEXT SUMMARY
    # =========================================================

    def print_summary(self, stats: PlaneStatistics) -> None:

        print("\n======================================")
        print("PLANE QUALITY REPORT")
        print("======================================")

        print(f"Frames:     {stats.total_frames}")
        print(f"Samples:    {stats.total_samples}")
        print(f"Detected:   {stats.total_detected}")
        print(f"Detection:  {stats.detection_rate:.2f}%")
        print(f"Points:     {stats.number_of_points}")

        print("\nBoard size (mm):")
        print(f"  Width:  {stats.board_width_mm:.2f}")
        print(f"  Height: {stats.board_height_mm:.2f}")

        print("\nPer-point summary:")

        for point in stats.statistics.values():

            print(
                f"  ({point.board_x_mm:.1f}, {point.board_y_mm:.1f}) "
                f"det={point.detection_rate:.1f}% "
                f"sharp={point.mean_sharpness:.1f} "
                f"size={point.mean_marker_size_px:.1f}"
            )
            