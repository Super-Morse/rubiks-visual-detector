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

### Step 1: Start local static server
Run the following command from the repository root:

```bash
python -m http.server 8080 --directory tools/dataset-generator
```
