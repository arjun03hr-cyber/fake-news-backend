from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
from app.utils import clean_text

app = FastAPI(title="Fake News Detection API")

# CORS (safe for frontend + Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- SAFE PATH HANDLING (CRITICAL) ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

class NewsRequest(BaseModel):
    text: str

# ---- HEALTH CHECK ENDPOINT ----
@app.get("/")
def root():
    return {"status": "Fake News Detection API running"}

# ---- MAIN PREDICTION ENDPOINT ----
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
