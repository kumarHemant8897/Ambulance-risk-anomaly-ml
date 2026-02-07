from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Optional

from artifact_detector import detect_signal_artifacts
from feature_extraction import extract_rolling_features
from isolation_forest_anomaly import train_model, predict
from risk_scoring import calculate_risk

app = FastAPI()

class VitalsData(BaseModel):
    timestamp: List[str]
    hr: List[float]
    spo2: List[float]
    bp_sys: List[float]
    bp_dia: List[float]
    motion: List[float]

class PredictionResponse(BaseModel):
    anomaly: int
    risk_score: float
    confidence: float

model = None
scaler = None

@app.post("/predict", response_model=PredictionResponse)
async def predict_vitals(data: VitalsData):
    global model, scaler
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data.timestamp),
        'hr': data.hr,
        'spo2': data.spo2,
        'bp_sys': data.bp_sys,
        'bp_dia': data.bp_dia,
        'motion': data.motion
    })
    
    df_with_artifacts = detect_signal_artifacts(df)
    
    features_df = extract_rolling_features(df_with_artifacts)
    
    if model is None or scaler is None:
        model, scaler = train_model(features_df.dropna())
    
    features_clean = features_df.fillna(features_df.mean())
    if features_clean.empty:
        return PredictionResponse(anomaly=1, risk_score=50.0, confidence=0.5)
    
    anomaly_predictions = predict(model, scaler, features_clean)
    
    last_anomaly = anomaly_predictions[-1] if len(anomaly_predictions) > 0 else 1
    
    hr_trend = features_df['hr_slope_30s'].iloc[-1] if 'hr_slope_30s' in features_df.columns else 0
    spo2_trend = features_df['spo2_slope_30s'].iloc[-1] if 'spo2_slope_30s' in features_df.columns else 0
    bp_trend = features_df['bp_sys_slope_30s'].iloc[-1] if 'bp_sys_slope_30s' in features_df.columns else 0
    
    artifact_flag = df_with_artifacts['artifact_flag'].iloc[-1] if len(df_with_artifacts) > 0 else False
    
    risk_score, confidence = calculate_risk(hr_trend, spo2_trend, bp_trend, artifact_flag)
    
    return PredictionResponse(
        anomaly=int(last_anomaly),
        risk_score=float(risk_score),
        confidence=float(confidence)
    )
