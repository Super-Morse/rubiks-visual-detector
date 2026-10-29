from functools import lru_cache

from app.domain.interfaces import IFaceDetector
from app.infrastructure.onnx_detector import YoloOnnxFaceDetector


@lru_cache
def get_face_detector() -> IFaceDetector:
    """Provides a singleton instance of the ONNX face detector."""
    return YoloOnnxFaceDetector()
