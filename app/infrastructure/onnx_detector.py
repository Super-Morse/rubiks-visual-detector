from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from app.domain.interfaces import IFaceDetector
from app.domain.models import BoundingBox, Color, DetectedSticker, FaceScanResult

# Mapping from YOLO class index to domain Color enum
CLASS_TO_COLOR: dict[int, Color] = {
    0: Color.WHITE,
    1: Color.YELLOW,
    2: Color.GREEN,
    3: Color.BLUE,
    4: Color.RED,
    5: Color.ORANGE,
}


class YoloOnnxFaceDetector(IFaceDetector):
    """ONNX Runtime-based detector for Rubik's cube stickers using YOLO11."""

    def __init__(
        self,
        model_path: str = "app/infrastructure/models/best.onnx",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> None:
        model_file = Path(model_path).resolve()
        if not model_file.is_file():
            raise FileNotFoundError(f"ONNX model file not found at: {model_file}")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 2
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_file),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image_bytes: bytes) -> FaceScanResult:
        """Decodes image, runs YOLO inference, and maps results to a 3x3 grid."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image from provided byte stream.")

        orig_h, orig_w = img.shape[:2]

        # 1. Preprocess: letterbox resize to 640x640 maintaining aspect ratio
        input_tensor, scale, pad_w, pad_h = self._preprocess(img, target_size=640)

        # 2. Inference: raw shape is (1, 10, 8400) for YOLO11
        outputs = self.session.run(None, {self.input_name: input_tensor})
        raw_output = outputs[0]

        # 3. Postprocess: threshold filtering and Non-Maximum Suppression (NMS)
        boxes, scores, class_ids = self._postprocess(
            raw_output=raw_output,
            orig_w=orig_w,
            orig_h=orig_h,
            scale=scale,
            pad_w=pad_w,
            pad_h=pad_h,
        )

        if len(boxes) < 9:
            raise RuntimeError(
                f"Detection failed: expected at least 9 stickers, but found {len(boxes)}."
            )

        # 4. If more than 9 stickers detected, pick the top 9 most confident detections
        top9_indices = np.argsort(scores)[::-1][:9]
        selected_boxes = [boxes[i] for i in top9_indices]
        selected_scores = [scores[i] for i in top9_indices]
        selected_classes = [class_ids[i] for i in top9_indices]

        # 5. Sort detections into standard 3x3 row-major grid order
        sorted_stickers = self._sort_into_grid(
            boxes=selected_boxes,
            scores=selected_scores,
            classes=selected_classes,
        )

        sticker_colors = [s.color for s in sorted_stickers]
        center_color = sorted_stickers[4].color  # Index 4 represents the center sticker

        return FaceScanResult(
            stickers=sticker_colors,
            details=sorted_stickers,
            center_color=center_color,
            is_valid_face=True,
        )

    def _preprocess(
        self, img: np.ndarray, target_size: int
    ) -> tuple[np.ndarray, float, float, float]:
        """Pads and resizes image to target dimensions, normalizing to [0.0, 1.0]."""
        h, w = img.shape[:2]
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        pad_w = (target_size - new_w) / 2
        pad_h = (target_size - new_h) / 2
        canvas[int(pad_h) : int(pad_h) + new_h, int(pad_w) : int(pad_w) + new_w] = resized

        # Convert BGR -> RGB, format to CHW, normalize to range [0, 1]
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, scale, pad_w, pad_h

    def _postprocess(
        self,
        raw_output: np.ndarray,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_w: float,
        pad_h: float,
    ) -> tuple[list[list[float]], list[float], list[int]]:
        """Extracts valid detections and projects bounding boxes back to original image coordinates."""
        # Transpose shape from (1, 10, 8400) to (8400, 10)
        predictions = np.squeeze(raw_output).T

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return [], [], []

        # Convert [center_x, center_y, width, height] to [top_left_x, top_left_y, width, height]
        cv_boxes: list[list[float]] = []
        for cx, cy, w, h in boxes_xywh:
            x1 = cx - w / 2
            y1 = cy - h / 2
            cv_boxes.append([float(x1), float(y1), float(w), float(h)])

        indices = cv2.dnn.NMSBoxes(
            bboxes=cv_boxes,
            scores=confidences.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )

        final_boxes: list[list[float]] = []
        final_scores: list[float] = []
        final_classes: list[int] = []

        if len(indices) > 0:
            for idx in indices.flatten():
                x1, y1, bw, bh = cv_boxes[idx]

                # Inverse letterbox projection to original image dimensions
                orig_x1 = (x1 - pad_w) / scale
                orig_y1 = (y1 - pad_h) / scale
                orig_x2 = (x1 + bw - pad_w) / scale
                orig_y2 = (y1 + bh - pad_h) / scale

                orig_x1 = max(0.0, min(float(orig_w), orig_x1))
                orig_y1 = max(0.0, min(float(orig_h), orig_y1))
                orig_x2 = max(0.0, min(float(orig_w), orig_x2))
                orig_y2 = max(0.0, min(float(orig_h), orig_y2))

                final_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
                final_scores.append(float(confidences[idx]))
                final_classes.append(int(class_ids[idx]))

        return final_boxes, final_scores, final_classes

    def _sort_into_grid(
        self,
        boxes: list[list[float]],
        scores: list[float],
        classes: list[int],
    ) -> list[DetectedSticker]:
        """Arranges 9 bounding boxes into 3 rows sorted from top to bottom, left to right."""
        items = []
        for i in range(len(boxes)):
            box = boxes[i]
            cy = (box[1] + box[3]) / 2.0
            cx = (box[0] + box[2]) / 2.0
            items.append({
                "box": box,
                "score": scores[i],
                "class_id": classes[i],
                "cx": cx,
                "cy": cy,
            })

        # Split into 3 rows along the Y axis
        items.sort(key=lambda item: item["cy"])
        row1 = sorted(items[0:3], key=lambda item: item["cx"])
        row2 = sorted(items[3:6], key=lambda item: item["cx"])
        row3 = sorted(items[6:9], key=lambda item: item["cx"])

        sorted_items = row1 + row2 + row3
        result: list[DetectedSticker] = []

        for idx, it in enumerate(sorted_items):
            b = it["box"]
            result.append(
                DetectedSticker(
                    color=CLASS_TO_COLOR[it["class_id"]],
                    confidence=round(it["score"], 4),
                    box=BoundingBox(
                        x_min=round(b[0], 2),
                        y_min=round(b[1], 2),
                        x_max=round(b[2], 2),
                        y_max=round(b[3], 2),
                    ),
                    grid_index=idx,
                )
            )

        return result
