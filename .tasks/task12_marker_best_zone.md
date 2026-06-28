Є програма quality\marker_quality.py
Треба створити подібну програму з назвою quality\marker_best_zone.py
Особливості:
1) програма пропонує переміщати групи з 9-ти маркерів зліва-направо зони із зсувом, який визначається як параметр конфігурації, наприклад, 10 мм
2) довжина зони визначається як параметр конфігурації, наприклад, 270 мм;
3) для кожної ітерації зсуву група маркерів розташовується за чотирма кутами (0, 90, 180, 270)
4) для кожного положення групи маркерів визначається Detection Rate для кожного маркера
5) для кожного положення групи маркерів визначається середній Detection Rate всіх маркерів групи
6) середні значення Detection Rate для кожного положення зберігати у файлі marker_zone.md 

Якщо виникнуть питання, задавай!

Є програма калібрування камери, яка містить наступні програмні модулі:
- https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/ChArUcoBoardGenerator.py?token=GHSAT0AAAAAADI4YE2HPUPHZMWWPBLZTPDM2SAV7IQ
- https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/calibrator.py?token=GHSAT0AAAAAADI4YE2GDDQN5P7F6SIAJYRQ2SAV7YA
- https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/camera.py?token=GHSAT0AAAAAADI4YE2H4GDX7KWMAITBGZL22SAWABQ
- https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/config.py?token=GHSAT0AAAAAADI4YE2H3GZ7AEAODIZVVHX62SAWALA
- https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/detectors.py?token=GHSAT0AAAAAADI4YE2GVAXJRP3XQKCK67K22SAWAUQ
- https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/main.py?token=GHSAT0AAAAAADI4YE2HIWXKAFHBFHAMSYM22SAWA4Q
- https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/quality_analyzer.py?token=GHSAT0AAAAAADI4YE2HQ3JVJP3JVLM3DBQA2SAWBFA
- https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/sample_collector.py?token=GHSAT0AAAAAADI4YE2HKQ3ADT3BRFEK27GW2SAWBRQ
- https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/ui.py?token=GHSAT0AAAAAADI4YE2GH2TSYI3RDYAZ2RB22SAWB2A

Програма може використовувати дошку з ArUco-маркерами.
Проаналізуй можливість, доцільність використання такої дошки для визначення зони якості роспізнавання маркерів.

Є програма, яка містить модулі:
https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/plane_quality/main_plane_quality.py
https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/plane_quality/plane_frame.py
https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/plane_quality/plane_quality_analyzer.py
https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/plane_quality/plane_quality_collector.py
https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/plane_quality/plane_quality_heatmap.py
https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/plane_quality/plane_quality_report.py
https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/plane_quality/plane_sample.py
https://raw.githubusercontent.com/oleksandrblazhko/ArUco-Markers-Experiments/refs/heads/main/calibration/plane_quality/plane_statistics.py

Ти можеш прочитати зміст модулів ?