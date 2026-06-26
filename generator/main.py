import os
import argparse
import sys

# Ensure generator package folder is in path if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parser import parse_marker_ids
from aruco_gen import get_dictionary, generate_marker_base64
from html_builder import build_print_html, calculate_layout

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "marker_ids.txt")
    default_output = os.path.join(script_dir, "markers_print.html")

    parser = argparse.ArgumentParser(
        description="Generate a print-ready A4 HTML grid of ArUco markers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        default=default_input,
        help="Path to the input file containing marker IDs (ranges like '0-100' or lists like '0,1,2')."
    )
    parser.add_argument(
        "-o", "--output",
        default=default_output,
        help="Path to the output HTML file."
    )
    parser.add_argument(
        "-s", "--marker-size",
        type=float,
        default=10.0,
        help="Size of each marker in millimeters."
    )
    parser.add_argument(
        "-b", "--border-size",
        type=float,
        default=1.0,
        help="Size of the white padding/border around each marker in millimeters."
    )
    parser.add_argument(
        "-m", "--margin",
        type=float,
        default=10.0,
        help="A4 page margins in millimeters."
    )
    parser.add_argument(
        "-d", "--dictionary",
        default="DICT_4X4_1000",
        help="ArUco dictionary name (e.g., DICT_4X4_1000, DICT_4X4_250, DICT_5X5_100, etc.)."
    )
    parser.add_argument(
        "--px-size",
        type=int,
        default=150,
        help="Pixel resolution for generating the raw marker image (keeps size small but sharp)."
    )

    args = parser.parse_args()

    # Create a default input file if it doesn't exist to help the user test easily
    if not os.path.exists(args.input):
        print(f"Input file '{args.input}' not found. Creating a default file with range '0-249'.")
        with open(args.input, 'w', encoding='utf-8') as f:
            f.write("# Default marker IDs configuration\n")
            f.write("# You can use ranges: 0-249\n")
            f.write("# Or comma-separated sequences: 0,1,2,3,4\n")
            f.write("0-249\n")

    try:
        # 1. Parse IDs from file
        print(f"Reading marker IDs from: {args.input}")
        marker_ids = parse_marker_ids(args.input)
        print(f"Parsed {len(marker_ids)} unique marker IDs.")
        
        if not marker_ids:
            print("Error: No marker IDs parsed from the configuration file.")
            sys.exit(1)

        # 2. Get ArUco dictionary
        print(f"Loading ArUco dictionary: {args.dictionary}")
        dictionary = get_dictionary(args.dictionary)

        # 3. Generate markers as Base64 strings
        print("Generating marker images and encoding to Base64...")
        markers_info = []
        for idx, m_id in enumerate(marker_ids):
            # Show progress for larger sets
            if (idx + 1) % 50 == 0 or (idx + 1) == len(marker_ids):
                print(f"  Generated {idx + 1}/{len(marker_ids)} markers...")
                
            b64_str = generate_marker_base64(m_id, dictionary, args.px_size)
            markers_info.append({
                "id": m_id,
                "b64": b64_str
            })

        # 4. Build HTML print sheet
        print("Calculating layout and building HTML page...")
        layout = calculate_layout(args.marker_size, args.border_size, args.margin)
        
        html_content = build_print_html(
            markers_info=markers_info,
            marker_size_mm=args.marker_size,
            border_size_mm=args.border_size,
            page_margin_mm=args.margin,
            dict_name=args.dictionary
        )

        # 5. Write output file
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print("\n=== Generation Successful ===")
        print(f"Output File:       {os.path.abspath(args.output)}")
        print(f"Grid Layout:       {layout['cols']} columns x {layout['rows']} rows")
        print(f"Markers per Page:  {layout['markers_per_page']}")
        total_pages = (len(marker_ids) + layout['markers_per_page'] - 1) // layout['markers_per_page']
        print(f"Total Pages:       {total_pages}")
        print("=============================\n")
        print("To print the markers, open the HTML file in any web browser (Chrome, Edge, Firefox) and press Ctrl+P.")
        print("Remember to set margins to 'None' in the print dialog for exact millimeter dimensions.")

    except Exception as e:
        print(f"Error occurred during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
