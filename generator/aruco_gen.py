import cv2
import base64
import numpy as np

def get_dictionary(dict_name: str):
    """
    Retrieves the OpenCV ArUco dictionary matching the name.
    Defaults to DICT_4X4_1000 if not found or empty.
    """
    if not dict_name:
        dict_name = "DICT_4X4_1000"
        
    # Normalize input
    dict_name = dict_name.upper().strip()
    if not dict_name.startswith("DICT_"):
        # Support shorthand like "4X4_1000" -> "DICT_4X4_1000"
        dict_name = "DICT_" + dict_name
        
    if hasattr(cv2.aruco, dict_name):
        dict_id = getattr(cv2.aruco, dict_name)
        # In newer OpenCV, getPredefinedDictionary takes an integer
        return cv2.aruco.getPredefinedDictionary(dict_id)
    else:
        # Fallback dictionary
        fallback_name = "DICT_4X4_1000"
        print(f"Warning: Dictionary '{dict_name}' not found. Using default '{fallback_name}'.")
        dict_id = getattr(cv2.aruco, fallback_name)
        return cv2.aruco.getPredefinedDictionary(dict_id)

def generate_marker_base64(marker_id: int, dictionary, size_px: int = 200) -> str:
    """
    Generates an ArUco marker image using OpenCV, encodes it to PNG,
    and returns its Base64-encoded string representation.
    """
    # Create marker image
    if hasattr(cv2.aruco, "generateImageMarker"):
        # OpenCV 4.7.0+
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    elif hasattr(cv2.aruco, "drawMarker"):
        # Older OpenCV versions
        marker_img = cv2.aruco.drawMarker(dictionary, marker_id, size_px)
    else:
        raise AttributeError("Your OpenCV version does not contain 'generateImageMarker' or 'drawMarker' in the cv2.aruco module.")
        
    # Encode to PNG in memory
    success, encoded_img = cv2.imencode('.png', marker_img)
    if not success:
        raise RuntimeError(f"Failed to encode ArUco marker {marker_id} to PNG.")
        
    # Convert to Base64
    base64_bytes = base64.b64encode(encoded_img.tobytes())
    return base64_bytes.decode('utf-8')
