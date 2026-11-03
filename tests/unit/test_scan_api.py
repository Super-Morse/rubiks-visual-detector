import io
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_cube_service, get_face_detector
from app.domain.interfaces import IFaceDetector
from app.domain.models import BoundingBox, Color, DetectedSticker, FaceScanResult
from app.infrastructure.validators import StandardCubeValidator
from app.main import app
from app.services.cube_service import CubeScannerService


class MockFaceDetector(IFaceDetector):
    """Mock detector returning valid white face for success path."""

    def detect(self, image_bytes: bytes) -> FaceScanResult:
        mock_stickers = [Color.WHITE] * 9
        mock_details = [
            DetectedSticker(
                color=Color.WHITE,
                confidence=0.99,
                box=BoundingBox(x_min=10.0, y_min=10.0, x_max=50.0, y_max=50.0),
                grid_index=i,
            )
            for i in range(9)
        ]
        return FaceScanResult(
            stickers=mock_stickers,
            details=mock_details,
            center_color=Color.WHITE,
            is_valid_face=True,
        )


@pytest.fixture
def client_with_mock_detector():
    mock_det = MockFaceDetector()
    app.dependency_overrides[get_face_detector] = lambda: mock_det
    app.dependency_overrides[get_cube_service] = lambda: CubeScannerService(
        detector=mock_det,
        validator=StandardCubeValidator(),
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_dummy_jpeg() -> io.BytesIO:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer.tobytes())


def test_scan_face_success(client_with_mock_detector: TestClient):
    response = client_with_mock_detector.post(
        "/api/v1/scan-face",
        files={"file": ("test.jpg", make_dummy_jpeg(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["stickers"]) == 9
    assert data["center_color"] == "white"
    assert data["is_valid_face"] is True


def test_scan_face_rejects_non_image(client_with_mock_detector: TestClient):
    text_content = io.BytesIO(b"Hello world")
    response = client_with_mock_detector.post(
        "/api/v1/scan-face",
        files={"file": ("note.txt", text_content, "text/plain")},
    )
    assert response.status_code == 400
    assert "Uploaded file must be a valid image format" in response.json()["detail"]


def test_scan_face_handles_value_error():
    """Hits lines 33-37 in endpoints.py (ValueError -> 422)."""
    mock_detector = MagicMock(spec=IFaceDetector)
    mock_detector.detect.side_effect = ValueError("Corrupted image byte stream")
    app.dependency_overrides[get_face_detector] = lambda: mock_detector

    client = TestClient(app)
    response = client.post(
        "/api/v1/scan-face",
        files={"file": ("test.jpg", make_dummy_jpeg(), "image/jpeg")},
    )
    assert response.status_code == 422
    assert "Corrupted image byte stream" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_scan_face_handles_runtime_error():
    """Hits lines 38-42 in endpoints.py (RuntimeError -> 400)."""
    mock_detector = MagicMock(spec=IFaceDetector)
    mock_detector.detect.side_effect = RuntimeError("Expected 9 stickers, found 3")
    app.dependency_overrides[get_face_detector] = lambda: mock_detector

    client = TestClient(app)
    response = client.post(
        "/api/v1/scan-face",
        files={"file": ("test.jpg", make_dummy_jpeg(), "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Expected 9 stickers, found 3" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_validate_cube_endpoint_success(client_with_mock_detector: TestClient):
    payload = {
        "faces": [
            {"face": "U", "stickers": ["white"] * 9},
            {"face": "R", "stickers": ["red"] * 9},
            {"face": "F", "stickers": ["green"] * 9},
            {"face": "D", "stickers": ["yellow"] * 9},
            {"face": "L", "stickers": ["orange"] * 9},
            {"face": "B", "stickers": ["blue"] * 9},
        ]
    }
    response = client_with_mock_detector.post("/api/v1/validate-cube", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert len(data["kociemba_string"]) == 54


def test_validate_cube_endpoint_invalid_distribution(client_with_mock_detector: TestClient):
    payload = {
        "faces": [
            {"face": "U", "stickers": ["white"] * 9},
            {"face": "R", "stickers": ["red"] * 9},
            {"face": "F", "stickers": ["green"] * 9},
            {"face": "D", "stickers": ["white"] + ["yellow"] * 8},
            {"face": "L", "stickers": ["orange"] * 9},
            {"face": "B", "stickers": ["blue"] * 9},
        ]
    }
    response = client_with_mock_detector.post("/api/v1/validate-cube", json=payload)
    assert response.status_code == 422
    assert "Each color must appear exactly 9 times" in response.json()["detail"]
