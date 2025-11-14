import tempfile
import os
import cv2
import base64
from . import main as spike_size_script

def analyze_size_from_bytes(image_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_input_file:
        temp_input_file.write(image_bytes)
        input_path = temp_input_file.name

    try:
        result_image_np, measurements = spike_size_script.process_image(input_path)
        
        if not measurements:
            raise ValueError("Measurement failed, no data returned.")
            
        _, buffer = cv2.imencode('.png', result_image_np)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        measurement_data = measurements[0]
        
        return {
            "width_cm": measurement_data["width_cm"],
            "height_cm": measurement_data["height_cm"],
            "image_base64": image_base64
        }
    except Exception as e:
        raise ValueError(f"SpikeSize script execution failed: {e}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)