from functools import lru_cache

from app.domain.interfaces import ICubeValidator, IFaceDetector
from app.infrastructure.onnx_detector import YoloOnnxFaceDetector
from app.infrastructure.validators import StandardCubeValidator
from app.services.cube_service import CubeScannerService


@lru_cache
def get_face_detector() -> IFaceDetector:
    """Provides a singleton instance of the ONNX face detector."""
    return YoloOnnxFaceDetector()


@lru_cache
def get_cube_validator() -> ICubeValidator:
    """Provides a singleton instance of the standard cube validator."""
    return StandardCubeValidator()


def get_cube_service() -> CubeScannerService:
    """Factory dependency providing CubeScannerService with injected components."""
    return CubeScannerService(
        detector=get_face_detector(),
        validator=get_cube_validator(),
    )
