from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_cube_service, get_face_detector
from app.api.schemas import CubeValidationSuccess, FullScanResponse, ValidateCubeRequest
from app.domain.exceptions import CubeDomainException
from app.domain.interfaces import IFaceDetector
from app.domain.models import FacePosition, FaceScanResult, FaceState
from app.services.cube_service import CubeScannerService

router = APIRouter()


@router.post(
    "/scan-face",
    response_model=FaceScanResult,
    status_code=status.HTTP_200_OK,
    summary="Scan a single cube face",
    description="Accepts an image and extracts 9 detected stickers arranged in 3x3 row-major order.",
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


@router.post(
    "/validate-cube",
    response_model=CubeValidationSuccess,
    status_code=status.HTTP_200_OK,
    summary="Validate full cube state",
    description="Validates 6 client-submitted faces and generates 54-char Kociemba string for the solver.",
)
async def validate_cube(
    payload: ValidateCubeRequest,
    service: CubeScannerService = Depends(get_cube_service),
) -> CubeValidationSuccess:
    try:
        faces_dict = {
            item.face: FaceState(face=item.face, stickers=item.stickers)
            for item in payload.faces
        }
        cube_state = service.assemble_and_validate(faces_dict)
        kociemba = cube_state.to_kociemba_string()

        return CubeValidationSuccess(
            is_valid=True,
            kociemba_string=kociemba,
            detected_faces={pos: state.stickers for pos, state in cube_state.faces.items()},
        )
    except CubeDomainException as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )


@router.post(
    "/scan-cube",
    response_model=FullScanResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch scan 6 cube photos",
    description="Accepts 6 face images, detects stickers, resolves faces by center color, and validates the cube.",
)
async def scan_full_cube(
    files: list[UploadFile] = File(...),
    service: CubeScannerService = Depends(get_cube_service),
) -> FullScanResponse:
    if len(files) != 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exactly 6 images are required, received {len(files)}.",
        )

    images_bytes: list[bytes] = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file.filename} is not a valid image.",
            )
        images_bytes.append(await file.read())

    try:
        cube_state, scan_details = service.process_six_images(images_bytes)
        kociemba = cube_state.to_kociemba_string()

        return FullScanResponse(
            is_valid=True,
            kociemba_string=kociemba,
            face_scans=scan_details,
        )
    except (CubeDomainException, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as run_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(run_err),
        )
