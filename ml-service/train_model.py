
from pathlib import Path
import json
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# =====================================
# 1. PROJECT PATHS
# =====================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "dataset" / "processed" / "train"
VAL_DIR = PROJECT_DIR / "dataset" / "processed" / "val"

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. TRAINING SETTINGS
# =====================================

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================
# 3. IMAGE TRANSFORMATIONS
# =====================================

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# 4. LOAD DATASET
# =====================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transforms
)

class_names = train_dataset.classes
num_classes = len(class_names)

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError("Train and validation classes do not match!")

print("Total Classes:", num_classes)
print("Training Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))


# =====================================
# 5. DATA LOADERS
# =====================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# =====================================
# 6. LOAD MOBILENETV2
# =====================================

print("\nLoading MobileNetV2...")

weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(weights=weights)

# Replace final classification layer
in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)

model = model.to(DEVICE)

print("MobileNetV2 Loaded Successfully!")


# =====================================
# 7. LOSS AND OPTIMIZER
# =====================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================
# 8. TRAINING FUNCTION
# =====================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy


# =====================================
# 9. VALIDATION FUNCTION
# =====================================

def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return val_loss, accuracy


# =====================================
# 10. START TRAINING
# =====================================

best_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())

print("\n========== TRAINING STARTED ==========")

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_weights = copy.deepcopy(model.state_dict())

        checkpoint = {
            "model_state_dict": best_weights,
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "model_name": "MobileNetV2"
        }

        torch.save(
            checkpoint,
            MODEL_DIR / "best_model.pth"
        )

        print("Best model saved!")


# =====================================
# 11. SAVE CLASS NAMES
# =====================================

with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

print("\n========== TRAINING COMPLETED ==========")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print("Model Location:", MODEL_DIR / "best_model.pth")
