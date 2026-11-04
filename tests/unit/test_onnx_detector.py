from pathlib import Path
import cv2
import numpy as np
import pytest

from app.domain.models import Color
from app.infrastructure.onnx_detector import YoloOnnxFaceDetector

REAL_MODEL_PATH = "app/infrastructure/models/best.onnx"


@pytest.fixture
def black_image_bytes() -> bytes:
    """Creates a pure black 640x640 JPEG (real YOLO finds 0 stickers on it)."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def tall_black_image_bytes() -> bytes:
    """Creates a vertical black image to test real letterbox math."""
    img = np.zeros((800, 400, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_init_raises_file_not_found():
    """Real file check: non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
        YoloOnnxFaceDetector(model_path="app/infrastructure/models/does_not_exist.onnx")


def test_init_success_with_real_model():
    """Real initialization of our trained YOLO11n ONNX model."""
    if not Path(REAL_MODEL_PATH).is_file():
        pytest.skip("best.onnx model file not found, skipping real inference test.")

    detector = YoloOnnxFaceDetector(
        model_path=REAL_MODEL_PATH,
        conf_threshold=0.5,
        iou_threshold=0.45,
    )

    assert detector.input_name == "images"
    assert detector.conf_threshold == 0.5
    assert detector.iou_threshold == 0.45


def test_detect_corrupted_bytes_raises_value_error():
    """Real decoding check: invalid byte stream raises ValueError."""
    if not Path(REAL_MODEL_PATH).is_file():
        pytest.skip("best.onnx model file not found.")

    detector = YoloOnnxFaceDetector(model_path=REAL_MODEL_PATH)
    with pytest.raises(ValueError, match="Unable to decode image"):
        detector.detect(b"this is not a valid jpeg or png")


def test_real_model_on_empty_image_returns_invalid_face(black_image_bytes):
    """Verifies that an image without cube stickers gracefully returns is_valid_face=False."""
    if not Path(REAL_MODEL_PATH).is_file():
        pytest.skip("best.onnx model file not found.")

    detector = YoloOnnxFaceDetector(model_path=REAL_MODEL_PATH)
    result = detector.detect(black_image_bytes)

    assert result.is_valid_face is False
    assert len(result.stickers) == 0
    assert result.center_color is None
    assert "Searching for cube face" in result.message


def test_detect_full_pipeline_and_grid_sorting(monkeypatch, black_image_bytes):
    """Tests 3x3 grid sorting and NMS using pytest monkeypatch with a dummy function."""
    if not Path(REAL_MODEL_PATH).is_file():
        pytest.skip("best.onnx model file not found.")

    detector = YoloOnnxFaceDetector(model_path=REAL_MODEL_PATH, conf_threshold=0.25)

    # Prepare a synthetic raw output tensor with 9 distinct stickers in unordered sequence
    synthetic_tensor = np.zeros((1, 10, 8400), dtype=np.float32)

    unordered_grid = [
        (300, 300, 0),  # Row 3, Col 3 -> grid_index 8 (White)
        (100, 100, 1),  # Row 1, Col 1 -> grid_index 0 (Yellow)
        (200, 100, 2),  # Row 1, Col 2 -> grid_index 1 (Green)
        (300, 100, 3),  # Row 1, Col 3 -> grid_index 2 (Blue)
        (100, 200, 4),  # Row 2, Col 1 -> grid_index 3 (Red)
        (200, 200, 5),  # Row 2, Col 2 -> grid_index 4 (Orange - Center)
        (300, 200, 0),  # Row 2, Col 3 -> grid_index 5 (White)
        (100, 300, 1),  # Row 3, Col 1 -> grid_index 6 (Yellow)
        (200, 300, 2),  # Row 3, Col 2 -> grid_index 7 (Green)
    ]

    for i, (cx, cy, class_id) in enumerate(unordered_grid):
        synthetic_tensor[0, 0, i] = cx
        synthetic_tensor[0, 1, i] = cy
        synthetic_tensor[0, 2, i] = 40.0
        synthetic_tensor[0, 3, i] = 40.0
        synthetic_tensor[0, 4 + class_id, i] = 0.95

    monkeypatch.setattr(detector.session, "run", lambda output_names, input_feed: [synthetic_tensor])

    result = detector.detect(black_image_bytes)

    assert result.is_valid_face is True
    assert len(result.stickers) == 9
    assert result.center_color == Color.ORANGE
    assert result.stickers[0] == Color.YELLOW

    for idx, sticker in enumerate(result.details):
        assert sticker.grid_index == idx


def test_detect_more_than_9_stickers_picks_top_9(monkeypatch, black_image_bytes):
    """Tests selecting exactly 9 stickers when >9 detections are returned."""
    if not Path(REAL_MODEL_PATH).is_file():
        pytest.skip("best.onnx model file not found.")

    detector = YoloOnnxFaceDetector(model_path=REAL_MODEL_PATH, conf_threshold=0.2)

    synthetic_tensor = np.zeros((1, 10, 8400), dtype=np.float32)

    # 12 boxes: 9 with confidence 0.99, 3 with confidence 0.30
    for i in range(12):
        synthetic_tensor[0, 0, i] = 100 + (i % 3) * 60
        synthetic_tensor[0, 1, i] = 100 + (i // 3) * 60
        synthetic_tensor[0, 2, i] = 40.0
        synthetic_tensor[0, 3, i] = 40.0
        synthetic_tensor[0, 4, i] = 0.99 if i < 9 else 0.30

    monkeypatch.setattr(detector.session, "run", lambda output_names, input_feed: [synthetic_tensor])

    result = detector.detect(black_image_bytes)
    assert len(result.stickers) == 9


def test_preprocess_tall_image_aspect_ratio(monkeypatch, tall_black_image_bytes):
    """Tests preprocessing and coordinate scaling on non-square portrait image."""
    if not Path(REAL_MODEL_PATH).is_file():
        pytest.skip("best.onnx model file not found.")

    detector = YoloOnnxFaceDetector(model_path=REAL_MODEL_PATH, conf_threshold=0.25)

    synthetic_tensor = np.zeros((1, 10, 8400), dtype=np.float32)
    for i in range(9):
        synthetic_tensor[0, 0, i] = 150 + (i % 3) * 50
        synthetic_tensor[0, 1, i] = 150 + (i // 3) * 50
        synthetic_tensor[0, 2, i] = 40.0
        synthetic_tensor[0, 3, i] = 40.0
        synthetic_tensor[0, 4, i] = 0.95

    monkeypatch.setattr(detector.session, "run", lambda output_names, input_feed: [synthetic_tensor])

    result = detector.detect(tall_black_image_bytes)
    assert result.is_valid_face is True
    assert len(result.stickers) == 9
