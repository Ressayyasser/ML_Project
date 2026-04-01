from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import time
from src.ml_predictor import get_ml_predictor

app = FastAPI(title="SmartCoach Pro — Exercise Classifier API")

class SensorPayload(BaseModel):
    acc_x: list[float]
    acc_y: list[float]
    acc_z: list[float]

@app.post("/predict")
def predict_exercise(payload: SensorPayload):
    t0 = time.perf_counter()
    predictor = get_ml_predictor()

    signal_df = pd.DataFrame({
        "acc_x": payload.acc_x,
        "acc_y": payload.acc_y,
        "acc_z": payload.acc_z,
    })

    exercise, confidence, prob_dict = predictor.predict(signal_df)
    latency_ms = (time.perf_counter() - t0) * 1000  

    return {
        "exercise": exercise,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "latency_ms": round(latency_ms, 2),
    }
