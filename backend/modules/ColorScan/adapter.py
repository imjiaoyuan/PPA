import tempfile
import os
from . import main as color_scan_script

def analyze_color_from_bytes(image_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_input_file:
        temp_input_file.write(image_bytes)
        input_path = temp_input_file.name

    try:
        r, g, b = color_scan_script.get_fruit_color(input_path)
        return {"r": r, "g": g, "b": b}
    except Exception as e:
        raise ValueError(f"ColorScan script execution failed: {e}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)