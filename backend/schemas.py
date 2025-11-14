from pydantic import BaseModel
from typing import List, Optional

class LeafGenderResponse(BaseModel):
    female_prob: float
    male_prob: float

class ColorScanResponse(BaseModel):
    r: int
    g: int
    b: int

class GrainMeasurement(BaseModel):
    berry_id: int
    width_cm: float
    height_cm: float

class GrainCountResponse(BaseModel):
    grain_count: int
    measurements: List[GrainMeasurement]
    image_base64: str

class SpikeSizeResponse(BaseModel):
    width_cm: float
    height_cm: float
    image_base64: str

class BranchAngleResponse(BaseModel):
    angle: float
    image_base64: str