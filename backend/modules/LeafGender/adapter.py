import tempfile
import os
from backend.modules.LeafGender import main as leaf_gender_script

def analyze_gender_from_bytes(image_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_input_file:
        temp_input_file.write(image_bytes)
        input_path = temp_input_file.name

    try:
        result = leaf_gender_script.predict_gender_from_path(input_path)
        return result
    except Exception as e:
        raise ValueError(f"LeafGender script execution failed: {e}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)