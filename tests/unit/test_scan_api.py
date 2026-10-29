import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_face_detector
from app.domain.interfaces import IFaceDetector
from app.domain.models import BoundingBox, Color, DetectedSticker, FaceScanResult
from app.infrastructure.onnx_detector import CLASS_TO_COLOR, YoloOnnxFaceDetector
from app.main import app


# ---------------------------------------------------------------------------
# Test Helpers & Mocks
# ---------------------------------------------------------------------------
class MockFaceDetector(IFaceDetector):
    def __init__(self, raise_error: Exception | None = None) -> None:
        self.raise_error = raise_error

    def detect(self, image_bytes: bytes) -> FaceScanResult:
        if self.raise_error:
            raise self.raise_error

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
def client():
    return TestClient(app)


def create_dummy_jpeg_bytes(width: int = 300, height: int = 200) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def generate_mock_onnx_output() -> np.ndarray:
    """Generates synthetic YOLO11 output (1, 10, 8400) with 10 detections."""
    raw = np.zeros((1, 10, 8400), dtype=np.float32)
    grid_coords = [
        (100, 100), (200, 100), (300, 100),
        (100, 200), (200, 200), (300, 200),
        (100, 300), (200, 300), (300, 300),
        (400, 400),  # 10th box to test top-9 slicing
    ]
    for i, (cx, cy) in enumerate(grid_coords):
        raw[0, 0, i] = cx
        raw[0, 1, i] = cy
        raw[0, 2, i] = 50.0
        raw[0, 3, i] = 50.0
        raw[0, 4 + (i % 6), i] = 0.95
    return raw


# ---------------------------------------------------------------------------
# 1. API Endpoints Tests (covers app/api/deps.py & app/api/v1/endpoints.py)
# ---------------------------------------------------------------------------
def test_deps_get_face_detector_returns_real_instance():
    detector = get_face_detector()
    assert isinstance(detector, YoloOnnxFaceDetector)


def test_scan_face_success(client):
    app.dependency_overrides[get_face_detector] = lambda: MockFaceDetector()
    try:
        response = client.post(
            "/api/v1/scan-face",
            files={"file": ("face.jpg", create_dummy_jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["stickers"]) == 9
        assert data["center_color"] == "white"
        assert data["is_valid_face"] is True
    finally:
        app.dependency_overrides.clear()


def test_scan_face_raises_422_on_value_error(client):
    error = ValueError("Cannot decode corrupt image bytes")
    app.dependency_overrides[get_face_detector] = lambda: MockFaceDetector(raise_error=error)
    try:
        response = client.post(
            "/api/v1/scan-face",
            files={"file": ("corrupt.jpg", create_dummy_jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 422
        assert "Cannot decode corrupt image bytes" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_scan_face_raises_400_on_runtime_error(client):
    error = RuntimeError("Detection failed: expected at least 9 stickers")
    app.dependency_overrides[get_face_detector] = lambda: MockFaceDetector(raise_error=error)
    try:
        response = client.post(
            "/api/v1/scan-face",
            files={"file": ("partial.jpg", create_dummy_jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 400
        assert "Detection failed: expected at least 9 stickers" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_scan_face_rejects_non_image_mimetype(client):
    response = client.post(
        "/api/v1/scan-face",
        files={"file": ("doc.txt", io.BytesIO(b"text"), "text/plain")},
    )
    assert response.status_code == 400
    assert "Uploaded file must be a valid image format" in response.json()["detail"]


def test_scan_face_rejects_missing_mimetype(client):
    response = client.post(
        "/api/v1/scan-face",
        files={"file": ("unknown", io.BytesIO(b"data"), "")},
    )
    assert response.status_code == 400
    assert "Uploaded file must be a valid image format" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 2. ONNX Detector Engine Tests (covers app/infrastructure/onnx_detector.py)
# ---------------------------------------------------------------------------
def test_onnx_init_raises_file_not_found():
    """Covers line 31 in onnx_detector.py"""
    with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
        YoloOnnxFaceDetector(model_path="non_existent_path.onnx")


def test_onnx_detect_raises_value_error_on_corrupt_bytes():
    """Covers lines 48-54 in onnx_detector.py"""
    detector = YoloOnnxFaceDetector()
    with pytest.raises(ValueError, match="Unable to decode image"):
        detector.detect(b"not_an_image")


def test_onnx_detect_raises_runtime_error_when_no_stickers():
    """Covers lines 68-71 in onnx_detector.py"""
    detector = YoloOnnxFaceDetector()
    blank_bytes = create_dummy_jpeg_bytes(width=100, height=100)
    with pytest.raises(RuntimeError, match="expected at least 9 stickers"):
        detector.detect(blank_bytes)


def test_onnx_postprocess_empty_when_no_predictions():
    """Covers line 147 (empty mask branch) in onnx_detector.py"""
    detector = YoloOnnxFaceDetector(conf_threshold=0.99)
    empty_raw = np.zeros((1, 10, 8400), dtype=np.float32)
    boxes, scores, classes = detector._postprocess(empty_raw, 640, 640, 1.0, 0.0, 0.0)
    assert boxes == []
    assert scores == []
    assert classes == []


def test_onnx_sort_into_grid_scrambled_boxes():
    """Covers lines 196-234 in onnx_detector.py"""
    detector = YoloOnnxFaceDetector()
    scrambled_boxes = [
        [45, 45, 55, 55],  # 8 (bottom-right)
        [5, 5, 15, 15],    # 0 (top-left)
        [25, 5, 35, 15],   # 1
        [45, 5, 55, 15],   # 2
        [5, 25, 15, 35],   # 3
        [25, 25, 35, 35],  # 4 (center)
        [45, 25, 55, 35],  # 5
        [5, 45, 15, 55],   # 6
        [25, 45, 35, 55],  # 7
    ]
    scores = [0.9] * 9
    classes = [0, 1, 2, 3, 4, 5, 0, 1, 2]
    sorted_items = detector._sort_into_grid(scrambled_boxes, scores, classes)
    assert len(sorted_items) == 9
    assert sorted_items[0].grid_index == 0
    assert sorted_items[4].grid_index == 4
    assert sorted_items[8].grid_index == 8


def test_onnx_detect_full_pipeline_mocked(monkeypatch):
    """Covers lines 56-93, 104-121, 134-187 in onnx_detector.py"""
    detector = YoloOnnxFaceDetector()
    mock_raw = generate_mock_onnx_output()
    monkeypatch.setattr(detector.session, "run", lambda names, feed_dict: [mock_raw])

    img_bytes = create_dummy_jpeg_bytes(width=640, height=640)
    result = detector.detect(img_bytes)

    assert isinstance(result, FaceScanResult)
    assert len(result.stickers) == 9
    assert len(result.details) == 9
    assert result.is_valid_face is True
    assert result.center_color == result.stickers[4]
