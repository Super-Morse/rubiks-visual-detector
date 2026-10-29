import cv2
import numpy as np
import pytest

from app.domain.models import Color, FaceScanResult
from app.infrastructure.onnx_detector import CLASS_TO_COLOR, YoloOnnxFaceDetector


def create_dummy_jpeg_bytes(width: int = 300, height: int = 200) -> bytes:
    """Creates valid in-memory dummy JPEG image bytes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def generate_mock_onnx_output(
    num_boxes: int = 10,
    conf: float = 0.95,
) -> np.ndarray:
    """Generates synthetic YOLO11 raw output with shape (1, 10, 8400).

    Channels:
    [0:4] -> cx, cy, w, h
    [4:10] -> 6 color class probabilities
    """
    raw = np.zeros((1, 10, 8400), dtype=np.float32)

    # 3x3 grid centers on a 640x640 canvas (plus one extra 10th box for top-9 trimming test)
    grid_coords = [
        (100, 100), (200, 100), (300, 100),
        (100, 200), (200, 200), (300, 200),
        (100, 300), (200, 300), (300, 300),
        (400, 400),  # 10th redundant box
    ]

    for i in range(min(num_boxes, len(grid_coords))):
        cx, cy = grid_coords[i]
        raw[0, 0, i] = cx       # center x
        raw[0, 1, i] = cy       # center y
        raw[0, 2, i] = 50.0     # width
        raw[0, 3, i] = 50.0     # height
        raw[0, 4 + (i % 6), i] = conf  # class score for one of 6 colors

    return raw


class TestYoloOnnxFaceDetector:
    def test_init_raises_file_not_found(self):
        """Verifies FileNotFoundError when ONNX file path is invalid."""
        with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
            YoloOnnxFaceDetector(model_path="non_existent_weights.onnx")

    def test_detect_raises_value_error_on_corrupt_bytes(self):
        """Verifies ValueError when input bytes cannot be decoded into an image."""
        detector = YoloOnnxFaceDetector()
        with pytest.raises(ValueError, match="Unable to decode image"):
            detector.detect(b"invalid_non_image_byte_stream")

    def test_detect_raises_runtime_error_when_under_9_stickers_found(self):
        """Verifies RuntimeError when detector finds fewer than 9 stickers."""
        detector = YoloOnnxFaceDetector()
        blank_image_bytes = create_dummy_jpeg_bytes()

        # Pure black image yields 0 detections
        with pytest.raises(RuntimeError, match="Detection failed: expected at least 9 stickers"):
            detector.detect(blank_image_bytes)

    def test_preprocess_generates_correct_tensor_and_padding(self):
        """Verifies letterbox scaling, padding, channel order, and normalization."""
        detector = YoloOnnxFaceDetector()
        dummy_img = np.zeros((300, 600, 3), dtype=np.uint8)

        tensor, scale, pad_w, pad_h = detector._preprocess(dummy_img, target_size=640)

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert 0.0 <= tensor.max() <= 1.0
        assert scale == 640 / 600
        assert pad_w >= 0.0
        assert pad_h >= 0.0

    def test_postprocess_empty_when_no_predictions_exceed_confidence(self):
        """Verifies early empty return when all prediction scores are below threshold."""
        detector = YoloOnnxFaceDetector(conf_threshold=0.8)
        # Raw tensor with zero confidences
        empty_raw_output = np.zeros((1, 10, 8400), dtype=np.float32)

        boxes, scores, class_ids = detector._postprocess(
            raw_output=empty_raw_output,
            orig_w=640,
            orig_h=640,
            scale=1.0,
            pad_w=0.0,
            pad_h=0.0,
        )

        assert boxes == []
        assert scores == []
        assert class_ids == []

    def test_sort_into_grid_orders_scrambled_stickers(self):
        """Verifies 3x3 geometric ordering from top-to-bottom, left-to-right."""
        detector = YoloOnnxFaceDetector()

        # Scrambled bounding boxes
        scrambled_boxes = [
            [45, 45, 55, 55],  # index 8: bottom-right
            [5, 5, 15, 15],    # index 0: top-left
            [25, 5, 35, 15],   # index 1: top-center
            [45, 5, 55, 15],   # index 2: top-right
            [5, 25, 15, 35],   # index 3: middle-left
            [25, 25, 35, 35],  # index 4: center
            [45, 25, 55, 35],  # index 5: middle-right
            [5, 45, 15, 55],   # index 6: bottom-left
            [25, 45, 35, 55],  # index 7: bottom-center
        ]
        scores = [0.9] * 9
        classes = [0, 1, 2, 3, 4, 5, 0, 1, 2]

        sorted_stickers = detector._sort_into_grid(scrambled_boxes, scores, classes)

        assert len(sorted_stickers) == 9
        for i, sticker in enumerate(sorted_stickers):
            assert sticker.grid_index == i

        # Verify key positions
        assert sorted_stickers[0].box.x_min == 5.0
        assert sorted_stickers[0].box.y_min == 5.0

        assert sorted_stickers[4].box.x_min == 25.0
        assert sorted_stickers[4].box.y_min == 25.0

        assert sorted_stickers[8].box.x_min == 45.0
        assert sorted_stickers[8].box.y_min == 45.0

    def test_detect_full_pipeline_with_mocked_session(self, monkeypatch):
        """Verifies the complete detection flow: decoding, inference, NMS, top-9 filter, grid sort."""
        detector = YoloOnnxFaceDetector()
        mock_raw_output = generate_mock_onnx_output(num_boxes=10, conf=0.98)

        # Mock session.run to avoid relying on external image patterns
        monkeypatch.setattr(detector.session, "run", lambda names, feed_dict: [mock_raw_output])

        test_img_bytes = create_dummy_jpeg_bytes(width=640, height=640)
        result = detector.detect(test_img_bytes)

        assert isinstance(result, FaceScanResult)
        assert len(result.stickers) == 9
        assert len(result.details) == 9
        assert result.is_valid_face is True
        assert result.center_color == result.stickers[4]
