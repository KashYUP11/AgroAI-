import os
import shutil
from pathlib import Path

# === CHANGE THESE TWO PATHS ===
SOURCE_ROOT = Path(r"C:\Users\Asus\Downloads\archive (1)\Plants_2")  # folder that contains train/ valid/ test/
DEST_ROOT = Path(r"C:\Users\Asus\PycharmProjects\LeafDisease-CNN\data\data_risk_kaggle")
# ==============================

HEALTHY_DIR = DEST_ROOT / "healthy"
AT_RISK_DIR = DEST_ROOT / "at_risk"

HEALTHY_DIR.mkdir(parents=True, exist_ok=True)
AT_RISK_DIR.mkdir(parents=True, exist_ok=True)

def copy_split(split_name: str):
    split_dir = SOURCE_ROOT / split_name
    print(f"\nProcessing split: {split_dir}")

    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower()
        if "healthy" in class_name:
            target_root = HEALTHY_DIR
        else:
            target_root = AT_RISK_DIR

        print(f"  Class '{class_dir.name}' -> {'healthy' if target_root is HEALTHY_DIR else 'at_risk'}")

        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            dest_path = target_root / f"{class_dir.name}_{img_path.name}"
            shutil.copy2(img_path, dest_path)

# Use train + valid splits
copy_split("train")
copy_split("valid")

print("\nDone!")
print(f"Healthy images: {len(list(HEALTHY_DIR.glob('*.*')))}")
print(f"At-risk images: {len(list(AT_RISK_DIR.glob('*.*')))}")
