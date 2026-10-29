import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_face_detector
from app.domain.interfaces import IFaceDetector
from app.domain.models import BoundingBox, Color, DetectedSticker, FaceScanResult
from app.main import app


class MockFaceDetector(IFaceDetector):
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
    app.dependency_overrides[get_face_detector] = lambda: MockFaceDetector()
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_scan_face_success(client_with_mock_detector):
    # Create a synthetic image in memory
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    file_bytes = io.BytesIO(buffer.tobytes())

    response = client_with_mock_detector.post(
        "/api/v1/scan-face",
        files={"file": ("test.jpg", file_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["stickers"]) == 9
    assert data["center_color"] == "white"
    assert data["is_valid_face"] is True


def test_scan_face_rejects_non_image(client_with_mock_detector):
    text_content = io.BytesIO(b"Hello world")
    response = client_with_mock_detector.post(
        "/api/v1/scan-face",
        files={"file": ("note.txt", text_content, "text/plain")},
    )
    assert response.status_code == 400
    assert "Uploaded file must be a valid image format" in response.json()["detail"]
