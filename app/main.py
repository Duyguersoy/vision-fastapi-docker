import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.inference import Predictor

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "7001"))

MODEL_PATH = os.getenv("MODEL_PATH", "models/detector.pth")
LE_PATH = os.getenv("LE_PATH", "models/le.pickle")
DEVICE = os.getenv("DEVICE", "cpu")

predictor: Optional[Predictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = Predictor(MODEL_PATH, LE_PATH, device=DEVICE)
    yield
    predictor = None

app = FastAPI(title="Vision Model Service", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    label, prob = predictor.predict_image_bytes(image_bytes)

    return {"label": label, "probability": prob}