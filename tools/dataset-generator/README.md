# Rubik's YOLO Dataset Studio

A browser-based synthetic data generator built with **Three.js** that produces perfectly annotated datasets for **YOLOv8 / YOLO11** object detection models without manual labeling.

---

## Key Features

* **3D-to-2D Matrix Projection**: Computes precise YOLO bounding boxes `(x_center, y_center, width, height)` mapped to camera viewport space `[0.0, 1.0]`.
* **Backface Culling**: Normal vector dot product verification ensures only visible stickers are labeled.
* **Domain Randomization**:
  * Full 3-axis rotation, scale, and distance variations.
  * Multi-directional light intensity and angle fluctuations.
  * Scrambled color configurations across all 54 stickers.
* **Automated Train/Val Split**: Automatically divides samples into an 80% Train / 20% Val split.
* **One-Click Export**: Bundles images, annotations, and `dataset.yaml` into a downloadable `.zip` archive.

---

## How to Run

1. **Start local static server**:
   Run the following command from the repository root:
   ```bash
   python -m http.server 8080 --directory tools/dataset-generator
   ```

2. **Open in browser**:
   Navigate to [http://localhost:8080](http://localhost:8080)

3. **Generate Dataset**:
   * Set the desired number of samples (e.g., 200 for testing, 1000 for production).
   * Select resolution (default: `640x640`).
   * Click **Generate & Export ZIP**.

---

## Dataset Structure

The downloaded ZIP archive is immediately compatible with Ultralytics YOLO:

```text
dataset/
├── dataset.yaml
├── images/
│   ├── train/      # 80% of generated images
│   └── val/        # 20% of generated images
└── labels/
    ├── train/      # Corresponding YOLO txt annotations
    └── val/        # Corresponding YOLO txt annotations
```

---

## Sim-to-Real Strategy (Adding Real Photos)

To bridge the gap between synthetic 3D rendering and real webcam footage:

1. **Capture**: Take 50 real photos of your physical Rubik's cube under various lighting conditions.
2. **Annotate**: Label these 50 images with bounding boxes (classes 0 to 5).
3. **Distribute**: Mix the real photos into the synthetic dataset to improve generalization:
   * Add 40 real photos + labels to `images/train/` and `labels/train/`.
   * Add 10 real photos + labels to `images/val/` and `labels/val/`.
4. **Train**: Train the YOLO model on this hybrid dataset to achieve 95%+ real-time detection accuracy.
