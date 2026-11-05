from app.api.deps import get_cube_service, get_cube_validator, get_face_detector
from app.domain.interfaces import ICubeValidator, IFaceDetector
from app.infrastructure.validators import StandardCubeValidator
from app.services.cube_service import CubeScannerService


def test_get_cube_validator_returns_singleton():
    """Validates that get_cube_validator returns StandardCubeValidator and caches instance."""
    validator_first = get_cube_validator()
    validator_second = get_cube_validator()

    assert isinstance(validator_first, ICubeValidator)
    assert isinstance(validator_first, StandardCubeValidator)
    assert validator_first is validator_second


def test_get_face_detector_returns_singleton(monkeypatch):
    """Validates that get_face_detector instantiates detector and respects lru_cache."""

    class DummyDetector(IFaceDetector):
        def detect(self, image_bytes: bytes):
            pass

    get_face_detector.cache_clear()
    monkeypatch.setattr("app.api.deps.YoloOnnxFaceDetector", DummyDetector)

    try:
        detector_first = get_face_detector()
        detector_second = get_face_detector()

        assert isinstance(detector_first, IFaceDetector)
        assert isinstance(detector_first, DummyDetector)
        assert detector_first is detector_second
    finally:
        get_face_detector.cache_clear()


def test_get_cube_service_wires_dependencies(monkeypatch):
    """Validates that get_cube_service injects detector and validator into CubeScannerService."""

    class DummyDetector(IFaceDetector):
        def detect(self, image_bytes: bytes):
            pass

    get_face_detector.cache_clear()
    monkeypatch.setattr("app.api.deps.YoloOnnxFaceDetector", DummyDetector)

    try:
        service = get_cube_service()

        assert isinstance(service, CubeScannerService)
        assert isinstance(service._detector, DummyDetector)
        assert isinstance(service._validator, StandardCubeValidator)
    finally:
        get_face_detector.cache_clear()
