from pathlib import Path
import logging
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ---------------------------------------
# Logging
# ---------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("mp219_api")


# ---------------------------------------
# Load model
# ---------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)


# ---------------------------------------
# FastAPI application
# ---------------------------------------

app = FastAPI(
    title="MP219 Canteen Prediction API",
    description="Predict canteen lunch plates sold.",
    version="1.0",
)


# ---------------------------------------
# Input schema
# ---------------------------------------

class PredictionInput(BaseModel):

    students_on_campus: int = Field(gt=0)
    menu_price_rs: float = Field(gt=0, le=200)
    temp_c: float = Field(ge=5)
    exam_days_left: int
    special_dishes: str
    queue_minutes_yesterday: float = Field(ge=0)
    day_type: str
    block: str


# ---------------------------------------
# Health endpoint
# ---------------------------------------

@app.get("/health")
def health():

    return {"status": "ok"}


# ---------------------------------------
# Prediction endpoint
# ---------------------------------------

@app.post("/predict")
def predict(data: PredictionInput):

    input_data = pd.DataFrame([{
        "students_on_campus": data.students_on_campus,
        "menu_price_rs": data.menu_price_rs,
        "temp_c": data.temp_c,
        "exam_days_left": data.exam_days_left,
        "special_dishes": data.special_dishes,
        "queue_minutes_yesterday": data.queue_minutes_yesterday,
        "day_type": data.day_type,
        "block": data.block,
    }])

    prediction = model.predict(input_data)[0]

    prediction = max(0, round(float(prediction)))

    logger.info(json.dumps({
        "event": "prediction",
        "prediction": prediction,
    }))

    return {
        "predicted_plates_sold": prediction
    }