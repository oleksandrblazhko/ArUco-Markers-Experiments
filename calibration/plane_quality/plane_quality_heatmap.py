"""
plane_quality_heatmap.py

Generates heatmaps from PlaneStatistics.

Supported maps:
- detection rate
- sharpness
- marker size
- distance from center
"""

import numpy as np
import matplotlib.pyplot as plt

from plane_statistics import PlaneStatistics


class PlaneQualityHeatmap:
    """
    Creates 2D heatmaps for plane quality analysis.
    """

    # =========================================================
    # INTERNAL: BUILD GRID
    # =========================================================

    def _build_grid(self, stats: PlaneStatistics):

        xs = [p.board_x_mm for p in stats.statistics.values()]
        ys = [p.board_y_mm for p in stats.statistics.values()]

        if not xs or not ys:
            return None, None, None

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        return x_min, x_max, y_min, y_max

    # =========================================================
    # INTERNAL: MAP VALUES
    # =========================================================

    def _extract(self, stats: PlaneStatistics, mode: str):

        xs = []
        ys = []
        values = []

        for p in stats.statistics.values():

            xs.append(p.board_x_mm)
            ys.append(p.board_y_mm)

            if mode == "detection":
                values.append(p.detection_rate)

            elif mode == "sharpness":
                values.append(p.mean_sharpness)

            elif mode == "size":
                values.append(p.mean_marker_size_px)

            elif mode == "distance":
                values.append(p.mean_distance_center_px)

            else:
                values.append(0.0)

        return np.array(xs), np.array(ys), np.array(values)

    # =========================================================
    # MAIN PLOT FUNCTION
    # =========================================================

    def plot(
        self,
        stats: PlaneStatistics,
        mode: str = "detection",
        title: str | None = None
    ):

        xs, ys, values = self._extract(stats, mode)

        if len(xs) == 0:
            print("No data for heatmap")
            return

        plt.figure(figsize=(8, 6))

        sc = plt.scatter(
            xs,
            ys,
            c=values,
            cmap="viridis",
            s=120,
            edgecolors="black"
        )

        plt.colorbar(sc)

        plt.xlabel("Board X (mm)")
        plt.ylabel("Board Y (mm)")

        if title is None:
            title = f"Plane Quality Heatmap ({mode})"

        plt.title(title)

        plt.gca().set_aspect("equal")

        plt.grid(True)

        plt.show()

    # =========================================================
    # SAVE TO FILE
    # =========================================================

    def save_png(
        self,
        stats: PlaneStatistics,
        filename: str,
        mode: str = "detection"
    ):

        xs, ys, values = self._extract(stats, mode)

        if len(xs) == 0:
            return

        plt.figure(figsize=(8, 6))

        sc = plt.scatter(
            xs,
            ys,
            c=values,
            cmap="viridis",
            s=120,
            edgecolors="black"
        )

        plt.colorbar(sc)

        plt.xlabel("Board X (mm)")
        plt.ylabel("Board Y (mm)")
        plt.title(f"Heatmap ({mode})")

        plt.gca().set_aspect("equal")

        plt.grid(True)

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        