
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from predict import load_model, transform


# =====================================
# 1. FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="AgriVision AI API",
    description="AI-powered plant disease detection",
    version="1.0.0"
)


# =====================================
# 2. CORS CONFIGURATION
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =====================================
# 3. LOAD MODEL
# =====================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model, class_names = load_model()

print("AgriVision AI Model Loaded!")
print("Device:", DEVICE)
print("Total Classes:", len(class_names))


# =====================================
# 4. HOME ROUTE
# =====================================

@app.get("/")
def home():

    return {
        "message": "Welcome to AgriVision AI API",
        "status": "Running",
        "model": "MobileNetV2",
        "device": str(DEVICE),
        "total_classes": len(class_names)
    }


# =====================================
# 5. HEALTH CHECK
# =====================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }


# =====================================
# 6. DISEASE PREDICTION
# =====================================

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must be less than 10 MB."
        )

    try:
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Unable to read image file."
        )

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    model.eval()

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

    confidence_score = round(
        confidence.item() * 100,
        2
    )

    return {
        "success": True,
        "predicted_disease": disease_name,
        "confidence": confidence_score,
        "device": str(DEVICE),
        "message": "Expert verification is recommended."
    }
