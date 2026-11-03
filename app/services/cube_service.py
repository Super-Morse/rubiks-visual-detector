from app.domain.exceptions import CubeDomainException, DuplicateCenterColorError
from app.domain.interfaces import ICubeValidator, IFaceDetector
from app.domain.models import (
    DEFAULT_COLOR_TO_FACE,
    CubeState,
    FacePosition,
    FaceScanResult,
    FaceState,
)


class CubeScannerService:
    """Orchestrates face detection, spatial mapping, and structural cube validation."""

    def __init__(self, detector: IFaceDetector, validator: ICubeValidator) -> None:
        self._detector = detector
        self._validator = validator

    def process_face_image(self, image_bytes: bytes) -> FaceScanResult:
        """Processes a single image frame and returns detected 3x3 stickers."""
        return self._detector.detect(image_bytes)

    def assemble_and_validate(self, face_states: dict[FacePosition, FaceState]) -> CubeState:
        """Assembles 6 faces into a CubeState and verifies structural integrity.

        Raises:
            CubeDomainException: If the cube violates physical 3x3 constraints.
        """
        cube = CubeState(faces=face_states)
        self._validator.validate(cube)
        return cube

    def process_six_images(self, images: list[bytes]) -> tuple[CubeState, dict[FacePosition, FaceScanResult]]:
        """Processes 6 images, auto-assigns faces by center color, and validates the cube.

        Raises:
            ValueError: If exactly 6 images are not provided.
            DuplicateCenterColorError: If multiple faces have the same center sticker color.
            CubeDomainException: If assembled cube state fails validation.
        """
        if len(images) != 6:
            raise ValueError(f"Expected exactly 6 images for a full cube scan, received {len(images)}.")

        detected_scans: list[FaceScanResult] = [
            self._detector.detect(img_bytes) for img_bytes in images
        ]

        faces_dict: dict[FacePosition, FaceState] = {}
        scan_details: dict[FacePosition, FaceScanResult] = {}

        for scan in detected_scans:
            center_col = scan.center_color
            target_pos = DEFAULT_COLOR_TO_FACE.get(center_col)

            if target_pos is None:
                raise ValueError(f"Unknown center color detected: {center_col}")

            if target_pos in faces_dict:
                raise DuplicateCenterColorError(
                    f"Duplicate face detected with center color: {center_col.value} (mapped to {target_pos.value})"
                )

            faces_dict[target_pos] = FaceState(face=target_pos, stickers=scan.stickers)
            scan_details[target_pos] = scan

        cube_state = self.assemble_and_validate(faces_dict)
        return cube_state, scan_details
