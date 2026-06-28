import re
import os

def parse_marker_ids(file_path: str) -> list[int]:
    """
    Reads a file containing marker IDs and parses them into a list of integers.
    Supports:
    - Ranges: e.g., "0-100" (returns IDs from 0 to 100 inclusive)
    - Sequences: e.g., "0,1,2,3" or space-separated "0 1 2 3"
    - Lines with comments starting with '#'
    - Mixed formats, e.g., multiple lines, some with ranges, some with lists
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Marker ID configuration file not found at: {file_path}")
        
    marker_ids = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip comments and whitespace
            line = line.split('#')[0].strip()
            if not line:
                continue
                
            # Split by commas or whitespace
            tokens = re.split(r'[\s,]+', line)
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                
                # Check for range format like "0-100" or "0:100"
                range_match = re.match(r'^(\d+)[\-:](\d+)$', token)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    # Ensure start is less than or equal to end
                    if start <= end:
                        marker_ids.extend(range(start, end + 1))
                    else:
                        marker_ids.extend(range(end, start + 1))
                else:
                    # Check for simple integer
                    try:
                        marker_ids.append(int(token))
                    except ValueError:
                        print(f"Warning: Skipping invalid token '{token}' in ID configuration file.")
                        
    # Return unique IDs preserving insertion order
    seen = set()
    unique_ids = []
    for x in marker_ids:
        if x not in seen:
            seen.add(x)
            unique_ids.append(x)
#    return unique_ids
    return marker_ids
