import pytest

from app.domain.exceptions import DuplicateCenterColorError
from app.domain.interfaces import IFaceDetector
from app.domain.models import (
    BoundingBox,
    Color,
    CubeState,
    DetectedSticker,
    FacePosition,
    FaceScanResult,
    FaceState,
)
from app.infrastructure.validators import StandardCubeValidator
from app.services.cube_service import CubeScannerService


class StubDetector(IFaceDetector):
    def __init__(self, results_queue: list[Color]) -> None:
        self.results_queue = results_queue
        self.call_count = 0

    def detect(self, image_bytes: bytes) -> FaceScanResult:
        center_color = self.results_queue[self.call_count]
        self.call_count += 1

        stickers = [center_color] * 9
        details = [
            DetectedSticker(
                color=center_color,
                confidence=0.98,
                box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
                grid_index=i,
            )
            for i in range(9)
        ]
        return FaceScanResult(
            stickers=stickers,
            details=details,
            center_color=center_color,
            is_valid_face=True,
        )


def make_solved_cube_dict() -> dict[FacePosition, FaceState]:
    return {
        FacePosition.U: FaceState(face=FacePosition.U, stickers=[Color.WHITE] * 9),
        FacePosition.R: FaceState(face=FacePosition.R, stickers=[Color.RED] * 9),
        FacePosition.F: FaceState(face=FacePosition.F, stickers=[Color.GREEN] * 9),
        FacePosition.D: FaceState(face=FacePosition.D, stickers=[Color.YELLOW] * 9),
        FacePosition.L: FaceState(face=FacePosition.L, stickers=[Color.ORANGE] * 9),
        FacePosition.B: FaceState(face=FacePosition.B, stickers=[Color.BLUE] * 9),
    }


def test_service_process_face_image():
    """Tests that process_face_image delegates frame processing to the underlying detector."""
    detector = StubDetector([Color.WHITE])
    service = CubeScannerService(detector=detector, validator=StandardCubeValidator())

    result = service.process_face_image(b"fake_image_bytes")

    assert result.is_valid_face is True
    assert result.center_color == Color.WHITE
    assert len(result.stickers) == 9


def test_service_assemble_and_validate_success():
    """Tests that 6 valid faces are properly assembled and validated into a CubeState."""
    service = CubeScannerService(
        detector=StubDetector([]),
        validator=StandardCubeValidator(),
    )
    faces = make_solved_cube_dict()
    cube = service.assemble_and_validate(faces)

    assert isinstance(cube, CubeState)
    assert cube.to_kociemba_string() == (
        "UUUUUUUUU"
        "RRRRRRRRR"
        "FFFFFFFFF"
        "DDDDDDDDD"
        "LLLLLLLLL"
        "BBBBBBBBB"
    )


def test_service_process_six_images_success():
    """Tests that 6 valid images are correctly processed and assigned to their respective faces."""
    colors = [Color.WHITE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.BLUE]
    detector = StubDetector(colors)
    service = CubeScannerService(detector=detector, validator=StandardCubeValidator())

    fake_images = [b"fake_image_bytes"] * 6
    cube, scan_details = service.process_six_images(fake_images)

    assert len(scan_details) == 6
    assert cube.faces[FacePosition.U].center_color == Color.WHITE
    assert cube.faces[FacePosition.R].center_color == Color.RED


def test_service_process_six_images_count_mismatch():
    """Tests that passing any number of images other than exactly 6 raises ValueError."""
    service = CubeScannerService(detector=StubDetector([]), validator=StandardCubeValidator())

    with pytest.raises(ValueError, match="Expected exactly 6 images"):
        service.process_six_images([b"fake"] * 5)


def test_service_process_six_images_rejects_duplicate_center():
    """Tests that passing duplicate center faces raises DuplicateCenterColorError."""
    colors = [Color.WHITE, Color.WHITE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.BLUE]
    detector = StubDetector(colors)
    service = CubeScannerService(detector=detector, validator=StandardCubeValidator())

    with pytest.raises(DuplicateCenterColorError):
        service.process_six_images([b"fake"] * 6)


def test_service_process_six_images_incomplete_face_raises_value_error(monkeypatch):
    """Tests that an incomplete face detection raises ValueError using monkeypatch."""
    detector = StubDetector([])
    incomplete_scan = FaceScanResult(
        stickers=[],
        details=[],
        center_color=None,
        is_valid_face=False,
    )
    monkeypatch.setattr(detector, "detect", lambda image_bytes: incomplete_scan)

    service = CubeScannerService(detector=detector, validator=StandardCubeValidator())

    with pytest.raises(ValueError, match="does not contain a complete 3x3 cube face"):
        service.process_six_images([b"fake"] * 6)


def test_service_process_six_images_unmapped_center_color(monkeypatch):
    """Tests that an unsupported center color raises ValueError using monkeypatch."""
    detector = StubDetector([])
    fake_scan = FaceScanResult(
        stickers=[Color.WHITE] * 9,
        details=[],
        center_color=Color.WHITE,
        is_valid_face=True,
    )
    monkeypatch.setattr(fake_scan, "center_color", "unsupported_color")
    monkeypatch.setattr(detector, "detect", lambda image_bytes: fake_scan)

    service = CubeScannerService(detector=detector, validator=StandardCubeValidator())

    with pytest.raises(ValueError, match="Unknown center color detected: unsupported_color"):
        service.process_six_images([b"fake"] * 6)
