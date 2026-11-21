
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# =====================================
# 1. PATHS AND DEVICE
# =====================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224


# =====================================
# 2. IMAGE TRANSFORMATION
# =====================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# 3. LOAD TRAINED MODEL
# =====================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, class_names


# =====================================
# 4. PREDICT DISEASE
# =====================================

def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model, class_names = load_model()

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    disease_name = class_names[
        predicted_class.item()
    ]

    confidence_score = confidence.item() * 100
    # print system and or printing 
    print("\n========== AGRIVISION AI ==========")

    print("Device:", DEVICE)

    print("Predicted Disease:", disease_name)

    print(f"Model Confidence: {confidence_score:.2f}%")

    print("==================================")

    if confidence_score < 60:
        print("Warning: Low confidence. Please verify the image.")

    elif confidence_score < 80:
        print("Result needs additional verification.")

    else:
        print("Prediction completed. Expert verification is recommended.")

    return disease_name, confidence_score


# =====================================
# 5. MAIN
# =====================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print('python .\\ml-service\\predict.py "image_path"')

        sys.exit(1)

    predict_image(sys.argv[1])
