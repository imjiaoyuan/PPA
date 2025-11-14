from fastapi import APIRouter, File, UploadFile, HTTPException, status
from backend.modules.SpikeSize import adapter
from backend.schemas import SpikeSizeResponse

router = APIRouter(
    prefix="/api/v1/spike-size",
    tags=["Spike Size"]
)

@router.post("/analyze", response_model=SpikeSizeResponse)
async def analyze_spike_size(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File provided is not an image."
        )
        
    image_bytes = await file.read()
    
    try:
        result = adapter.analyze_size_from_bytes(image_bytes)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )