import math

def calculate_layout(marker_size_mm: float, border_size_mm: float, page_margin_mm: float) -> dict:
    """
    Calculates how many markers can fit on a standard A4 page (210mm x 297mm)
    based on the marker size, white border size, and page margins.
    
    Returns a dictionary with layout parameters.
    """
    page_width_mm = 210.0
    page_height_mm = 297.0
    
    # Usable area inside page margins
    usable_width_mm = page_width_mm - (2.0 * page_margin_mm)
    usable_height_mm = page_height_mm - (2.0 * page_margin_mm)
    
    # A single cell contains the marker, two white paddings, and thin borders on both sides.
    # We assume a border thickness of 0.05mm.
    border_thickness_mm = 0.05
    cell_outer_width = marker_size_mm + (2.0 * border_size_mm) + (2.0 * border_thickness_mm)
    cell_outer_height = marker_size_mm + (2.0 * border_size_mm) + (2.0 * border_thickness_mm)
    
    # Calculate columns and rows that can fit
    cols = int(usable_width_mm // cell_outer_width)
    rows = int(usable_height_mm // cell_outer_height)
    
    # Ensure at least 1 column and row
    cols = max(1, cols)
    rows = max(1, rows)
    
    markers_per_page = cols * rows
    
    return {
        "cols": cols,
        "rows": rows,
        "markers_per_page": markers_per_page,
        "cell_outer_width": cell_outer_width,
        "cell_outer_height": cell_outer_height,
        "usable_width_mm": usable_width_mm,
        "usable_height_mm": usable_height_mm
    }

def build_print_html(markers_info: list[dict], marker_size_mm: float, border_size_mm: float, page_margin_mm: float, dict_name: str) -> str:
    """
    Generates a print-ready HTML page with the markers arranged in a grid.
    markers_info should be a list of dicts, e.g., [{"id": 0, "b64": "..."}].
    """
    layout = calculate_layout(marker_size_mm, border_size_mm, page_margin_mm)
    cols = layout["cols"]
    rows = layout["rows"]
    markers_per_page = layout["markers_per_page"]
    
    # Split markers into pages
    pages = []
    for i in range(0, len(markers_info), markers_per_page):
        pages.append(markers_info[i:i + markers_per_page])
        
    # Generate HTML content
    html_lines = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html lang=\"en\">")
    html_lines.append("<head>")
    html_lines.append("    <meta charset=\"UTF-8\">")
    html_lines.append("    <title>ArUco Markers Print Sheet</title>")
    html_lines.append("    <style>")
    html_lines.append("        /* Print styling rules */")
    html_lines.append("        @media print {")
    html_lines.append("            body {")
    html_lines.append("                margin: 0;")
    html_lines.append("                padding: 0;")
    html_lines.append("                background-color: white;")
    html_lines.append("            }")
    html_lines.append("            .page {")
    html_lines.append("                margin: 0 !important;")
    html_lines.append("                box-shadow: none !important;")
    html_lines.append("                page-break-after: always;")
    html_lines.append("            }")
    html_lines.append("            .page:last-child {")
    html_lines.append("                page-break-after: avoid;")
    html_lines.append("            }")
    html_lines.append("        }")
    html_lines.append("        @page {")
    html_lines.append("            size: A4 portrait;")
    html_lines.append("            margin: 0;")
    html_lines.append("        }")
    html_lines.append("        ")
    html_lines.append("        /* Screen preview styling */")
    html_lines.append("        body {")
    html_lines.append("            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;")
    html_lines.append("            margin: 0;")
    html_lines.append("            padding: 20px;")
    html_lines.append("            background-color: #f5f6fa;")
    html_lines.append("            display: flex;")
    html_lines.append("            flex-direction: column;")
    html_lines.append("            align-items: center;")
    html_lines.append("        }")
    html_lines.append("        .header {")
    html_lines.append("            background-color: white;")
    html_lines.append("            padding: 15px 25px;")
    html_lines.append("            border-radius: 8px;")
    html_lines.append("            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);")
    html_lines.append("            margin-bottom: 20px;")
    html_lines.append("            width: calc(210mm - 50px);")
    html_lines.append("        }")
    html_lines.append("        .header h1 {")
    html_lines.append("            margin: 0 0 10px 0;")
    html_lines.append("            font-size: 1.5rem;")
    html_lines.append("            color: #2f3640;")
    html_lines.append("        }")
    html_lines.append("        .header p {")
    html_lines.append("            margin: 3px 0;")
    html_lines.append("            font-size: 0.9rem;")
    html_lines.append("            color: #718093;")
    html_lines.append("        }")
    html_lines.append("        ")
    html_lines.append("        /* A4 Page layout container */")
    html_lines.append("        .page {")
    html_lines.append(f"            width: 210mm;")
    html_lines.append(f"            height: 297mm;")
    html_lines.append("            box-sizing: border-box;")
    html_lines.append("            background-color: white;")
    html_lines.append("            margin: 10px auto 30px auto;")
    html_lines.append(f"            padding: {page_margin_mm}mm;")
    html_lines.append("            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);")
    html_lines.append("            display: flex;")
    html_lines.append("            flex-wrap: wrap;")
    html_lines.append("            align-content: flex-start;")
    html_lines.append("            justify-content: flex-start;")
    html_lines.append("            overflow: hidden;")
    html_lines.append("        }")
    html_lines.append("        ")
    html_lines.append("        /* Individual Marker container */")
    html_lines.append("        .marker-cell {")
    html_lines.append(f"            width: {marker_size_mm}mm;")
    html_lines.append(f"            height: {marker_size_mm}mm;")
    html_lines.append(f"            padding: {border_size_mm}mm;")
    html_lines.append("            border: 0.05mm solid #F0F0F0; /* Thin gray separation line */")
    html_lines.append("            margin: -0.025mm; /* Collapse borders to prevent double thickness */")
    html_lines.append("            display: flex;")
    html_lines.append("            justify-content: center;")
    html_lines.append("            align-items: center;")
    html_lines.append("            background-color: #ffffff;")
    html_lines.append("            box-sizing: content-box;")
    html_lines.append("        }")
    html_lines.append("        ")
    html_lines.append("        .marker-img {")
    html_lines.append(f"            width: {marker_size_mm}mm;")
    html_lines.append(f"            height: {marker_size_mm}mm;")
    html_lines.append("            display: block;")
    html_lines.append("            image-rendering: pixelated; /* Keeps ArUco squares sharp */")
    html_lines.append("            image-rendering: crisp-edges;")
    html_lines.append("        }")
    html_lines.append("        ")
    html_lines.append("        /* Button to print from browser */")
    html_lines.append("        .print-btn {")
    html_lines.append("            background-color: #4cd137;")
    html_lines.append("            color: white;")
    html_lines.append("            border: none;")
    html_lines.append("            padding: 8px 16px;")
    html_lines.append("            border-radius: 4px;")
    html_lines.append("            cursor: pointer;")
    html_lines.append("            font-weight: bold;")
    html_lines.append("            font-size: 0.9rem;")
    html_lines.append("            margin-top: 10px;")
    html_lines.append("            transition: background-color 0.2s;")
    html_lines.append("        }")
    html_lines.append("        .print-btn:hover {")
    html_lines.append("            background-color: #44bd32;")
    html_lines.append("        }")
    html_lines.append("    </style>")
    html_lines.append("</head>")
    html_lines.append("<body>")
     
    # Render Pages
    for page_idx, page_markers in enumerate(pages):
        html_lines.append(f"    <div class=\"page\" id=\"page-{page_idx+1}\">")
        for m in page_markers:
            html_lines.append(f"        <div class=\"marker-cell\" title=\"ID: {m['id']}\">")
            html_lines.append(f"            <img class=\"marker-img\" src=\"data:image/png;base64,{m['b64']}\" alt=\"ArUco Marker {m['id']}\">")
            html_lines.append("        </div>")
        html_lines.append("    </div>")
        
    html_lines.append("</body>")
    html_lines.append("</html>")
    
    return "\n".join(html_lines)
