from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
from app.utils import clean_text

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="Fake News Detection API",
    version="0.1.0"
)

# --------------------------------------------------
# CORS (Safe for frontend + Render)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Base directory (VERY IMPORTANT for Render)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# --------------------------------------------------
# Load ML artifacts ONCE at startup
# --------------------------------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# --------------------------------------------------
# Request Schema
# --------------------------------------------------
class NewsRequest(BaseModel):
    text: str

# --------------------------------------------------
# Health Check / Root Endpoint
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "Fake News Detection API running"}

# --------------------------------------------------
# Main Prediction Endpoint
# --------------------------------------------------
@app.post("/analyze")
def analyze_news(request: NewsRequest):
    cleaned_text = clean_text(request.text)
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()

    return {
        "verdict": "Fake" if prediction == 1 else "Real",
        "confidence": round(float(confidence), 2)
    }
