"""
YOLO11 Training & ONNX Export Pipeline for Rubik's Cube detection.
Trains on the unified dataset and exports an optimized ONNX model for production deployment.
"""

from pathlib import Path
from ultralytics import YOLO


def train_and_export(
    data_yaml: str = "dataset/data.yaml",
    model_name: str = "yolo11n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
) -> None:
    """Train YOLO11 model on the unified Rubik's dataset and export to ONNX."""
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")

    print(f"Loading pretrained backbone: {model_name}...")
    model = YOLO(model_name)

    print(f"Starting training on {yaml_path} for {epochs} epochs...")
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/detect",
        name="rubiks_detector",
        save=True,
        plots=True,
    )

    print("\nTraining completed! Evaluating validation metrics...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")

    print("\nExporting model to optimized ONNX format for production Docker deployment...")
    exported_path = model.export(format="onnx", dynamic=False, simplify=True)
    print(f"Successfully exported ONNX model to: {exported_path}")


if __name__ == "__main__":
    train_and_export()
