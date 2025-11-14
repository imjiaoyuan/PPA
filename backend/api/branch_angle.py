from fastapi import APIRouter, File, UploadFile, HTTPException, status
from backend.modules.BranchAngle import adapter
from backend.schemas import BranchAngleResponse

router = APIRouter(
    prefix="/api/v1/branch-angle",
    tags=["Branch Angle"]
)

@router.post("/analyze", response_model=BranchAngleResponse)
async def analyze_branch_angle(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File provided is not an image."
        )
        
    image_bytes = await file.read()
    
    try:
        result = adapter.analyze_angle_from_bytes(image_bytes)
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