
from pathlib import Path
import random
import shutil

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    BASE_DIR / "dataset" / "plantvillage"
    / "Plant_leave_diseases_dataset_without_augmentation"
)

OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"

# Settings
TRAIN_RATIO = 0.80
SEED = 42

random.seed(SEED)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE_DIR}")

# Create output folders
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

total_train = 0
total_val = 0
total_classes = 0

for class_dir in sorted(SOURCE_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    # Exclude background images
    if class_dir.name == "Background_without_leaves":
        continue

    images = [
        file for file in class_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if len(images) < 2:
        print(f"Skipping: {class_dir.name}")
        continue

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_class_dir = TRAIN_DIR / class_dir.name
    val_class_dir = VAL_DIR / class_dir.name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    for image in train_images:
        shutil.copy2(
            image,
            train_class_dir / image.name
        )

    for image in val_images:
        shutil.copy2(
            image,
            val_class_dir / image.name
        )

    total_train += len(train_images)
    total_val += len(val_images)
    total_classes += 1

    print(
        f"{class_dir.name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(val_images)}"
    )

    # printing file sytem 

print("\n========== DATASET SPLIT COMPLETE ==========")
print("Total Classes:", total_classes)
print("Training Images:", total_train)
print("Validation Images:", total_val)
print("Total Images:", total_train + total_val)
print("Output Directory:", OUTPUT_DIR)
