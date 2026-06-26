import numpy as np
import cv2

class QualityAnalyzer:
    def __init__(self, image_width, image_height, grid_size=(3, 3)):
        self.image_width = image_width
        self.image_height = image_height
        self.grid_size = grid_size
        
        self.centers = []
        self.areas = []
        self.aspect_ratios = []
        self.grid_visits = set()

    def add_sample(self, corners):
        # Center
        center_x = np.mean(corners[:, 0, 0])
        center_y = np.mean(corners[:, 0, 1])
        self.centers.append((center_x, center_y))

        # Area
        area = cv2.contourArea(corners)
        image_area = self.image_width * self.image_height
        self.areas.append(area / image_area)

        # Aspect Ratio (for tilt)
        rect = cv2.minAreaRect(corners)
        w, h = rect[1]
        if w > 0 and h > 0:
            self.aspect_ratios.append(min(w, h) / max(w, h))

        # Grid Visit
        cell_x = int(center_x / (self.image_width / self.grid_size[0]))
        cell_y = int(center_y / (self.image_height / self.grid_size[1]))
        self.grid_visits.add((cell_x, cell_y))

    def generate_report(self, reprojection_error):
        if not self.centers:
            print("No samples to analyze.")
            return

        xs = [c[0] for c in self.centers]
        ys = [c[1] for c in self.centers]
        
        coverage_x = (max(xs) - min(xs)) / self.image_width
        coverage_y = (max(ys) - min(ys)) / self.image_height

        min_size = min(self.areas)
        max_size = max(self.areas)

        tilt_diversity = np.std(self.aspect_ratios) if self.aspect_ratios else 0

        grid_coverage = len(self.grid_visits)
        total_grid_cells = self.grid_size[0] * self.grid_size[1]

        # Determine overall quality and recommendations
        overall = "GOOD"
        recommendations = []
        if coverage_x < 0.7 or coverage_y < 0.7:
            overall = "POOR"
            recommendations.append("Increase board movement to cover more of the screen.")
        if max_size - min_size < 0.5:
            overall = "POOR"
            recommendations.append("Vary the distance of the board from the camera.")
        if tilt_diversity < 0.1:
            overall = "POOR"
            recommendations.append("Capture more tilted views of the board.")
        if grid_coverage < total_grid_cells:
            if overall == "GOOD":
                overall = "ACCEPTABLE"
            recommendations.append("Ensure the board visits all regions of the screen.")

        print("\n=== CALIBRATION QUALITY REPORT ===")
        print(f"\nReprojection error: {reprojection_error:.4f} px")
        
        print("\nCoverage:")
        print(f"  X coverage: {coverage_x:.1%}")
        print(f"  Y coverage: {coverage_y:.1%}")

        print("\nBoard scale:")
        print(f"  Min size: {min_size:.1%}")
        print(f"  Max size: {max_size:.1%}")

        print("\nOrientation diversity:")
        print(f"  Tilt diversity score: {tilt_diversity:.3f}")

        print("\nGrid coverage:")
        print(f"  {grid_coverage} / {total_grid_cells} regions visited")

        print(f"\nOverall: {overall}")
        if recommendations:
            print("\nRecommendation(s):")
            for rec in recommendations:
                print(f"- {rec}")

