from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "dataset"
    / "plantvillage"
    / "Plant_leave_diseases_dataset_without_augmentation"
)

# ftydty
total_images = 0
total_classes = 0

for folder in sorted(DATASET_DIR.iterdir()):
    if not folder.is_dir():
        continue

    if folder.name == "Background_without_leaves":
        continue

    count = sum(
        1 for file in folder.rglob("*")
        if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    total_classes += 1
    total_images += count

    print(f"{folder.name}: {count}")

print("\nTotal Classes:", total_classes)
print("Total Images:", total_images)
print("Dataset inspection completed!")
