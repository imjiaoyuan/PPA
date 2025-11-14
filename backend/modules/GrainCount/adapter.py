import tempfile
import os
import cv2
import base64
from backend.modules.GrainCount import main as grain_count_script

def analyze_grains_from_bytes(image_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_input_file:
        temp_input_file.write(image_bytes)
        input_path = temp_input_file.name

    try:
        result_image_np, measurements = grain_count_script.process_image(input_path)
        
        _, buffer = cv2.imencode('.png', result_image_np)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "grain_count": len(measurements),
            "measurements": measurements,
            "image_base64": image_base64
        }
    except Exception as e:
        raise ValueError(f"GrainCount script execution failed: {e}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)