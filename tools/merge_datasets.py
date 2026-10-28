"""
Dataset Merger & Class Normalizer for YOLO datasets.
Standardizes class labels and merges multiple datasets into one unified dataset.
"""

from pathlib import Path
import shutil
import yaml

# Standard unified classes for our project
CANONICAL_CLASSES = [
    "white",
    "yellow",
    "green",
    "blue",
    "red",
    "orange",
    "rubik_cube",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}


def normalize_class_name(name: str) -> str:
    """Normalize class names to standard lowercase canonical names."""
    clean = name.strip().lower().replace(" ", "_").replace("-", "_")
    if clean in ["cube", "rubik", "rubik_cube", "rubikscube", "rubikcube"]:
        return "rubik_cube"
    return clean


def process_dataset(src_dir: Path, out_dir: Path) -> None:
    """Read a YOLO dataset, remap its class indices, and append to output directory."""
    yaml_path = src_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"Skipping {src_dir}: data.yaml not found")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    # Resolve class names mapping from source data.yaml
    src_names = meta.get("names", [])
    if isinstance(src_names, dict):
        src_map = {int(k): normalize_class_name(v) for k, v in src_names.items()}
    else:
        src_map = {i: normalize_class_name(v) for i, v in enumerate(src_names)}

    print(f"Loaded {src_dir.name} with source classes: {src_map}")

    # Process both train and valid / val splits
    split_mappings = [
        ("train", "train"),
        ("valid", "val"),
        ("val", "val"),
    ]

    for src_split, target_split in split_mappings:
        src_img_dir = src_dir / src_split / "images"
        src_lbl_dir = src_dir / src_split / "labels"

        if not src_img_dir.exists():
            continue

        target_img_dir = out_dir / "images" / target_split
        target_lbl_dir = out_dir / "labels" / target_split
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_file in src_img_dir.glob("*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Prefix file name with dataset source name to avoid name collisions
            base_name = f"{src_dir.name}_{img_file.stem}"
            dest_img = target_img_dir / f"{base_name}{img_file.suffix}"
            shutil.copy2(img_file, dest_img)

            # Process matching label file
            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                remapped_lines = []
                with open(lbl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        old_cls_id = int(parts[0])
                        class_name = src_map.get(old_cls_id)

                        if class_name in CLASS_TO_ID:
                            new_cls_id = CLASS_TO_ID[class_name]
                            remapped_lines.append(f"{new_cls_id} {' '.join(parts[1:])}")

                dest_lbl = target_lbl_dir / f"{base_name}.txt"
                with open(dest_lbl, "w", encoding="utf-8") as f:
                    f.write("\n".join(remapped_lines) + "\n")


def main():
    raw_dir = Path("raw_data")
    output_dir = Path("dataset")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all subdirectories inside raw_data
    for dataset_folder in sorted(raw_dir.iterdir()):
        if dataset_folder.is_dir():
            print(f"\nProcessing {dataset_folder.name}...")
            process_dataset(dataset_folder, output_dir)

    # Create the unified master data.yaml
    master_yaml = {
        "path": "../dataset",
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(CANONICAL_CLASSES)},
    }

    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(master_yaml, f, sort_keys=False)

    # Print summary
    train_count = len(list((output_dir / "images" / "train").glob("*.*")))
    val_count = len(list((output_dir / "images" / "val").glob("*.*")))
    print(f"\nSuccessfully unified dataset into '{output_dir}'!")
    print(f"Total Train images: {train_count}")
    print(f"Total Val images:   {val_count}")
    print(f"Canonical classes:  {CANONICAL_CLASSES}")


if __name__ == "__main__":
    main()
