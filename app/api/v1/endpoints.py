from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.api.deps import get_face_detector
from app.domain.interfaces import IFaceDetector
from app.domain.models import FaceScanResult

router = APIRouter()


@router.post(
    "/scan-face",
    response_model=FaceScanResult,
    status_code=status.HTTP_200_OK,
    summary="Scan a single cube face",
    description="Accepts an image file and extracts 9 detected stickers arranged in 3x3 row-major order.",
)
async def scan_cube_face(
    file: UploadFile,
    detector: IFaceDetector = Depends(get_face_detector),
) -> FaceScanResult:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image format (JPEG, PNG, WebP).",
        )

    try:
        image_bytes = await file.read()
        return detector.detect(image_bytes)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except RuntimeError as run_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(run_err),
        )
