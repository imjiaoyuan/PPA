import tempfile
import os
import cv2
import base64
from backend.modules.BranchAngle import main as branch_angle_script

def analyze_angle_from_bytes(image_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_input_file:
        temp_input_file.write(image_bytes)
        input_path = temp_input_file.name

    try:
        angle_value, result_image_np = branch_angle_script.process_image(input_path)
        
        _, buffer = cv2.imencode('.png', result_image_np)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {"angle": angle_value, "image_base64": image_base64}
    except Exception as e:
        raise ValueError(f"BranchAngle script execution failed: {e}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)