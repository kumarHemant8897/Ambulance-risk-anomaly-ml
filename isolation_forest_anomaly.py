import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def train_model(X, contamination=0.1, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    
    model.fit(X_scaled)
    
    return model, scaler

def predict(model, scaler, X):
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions

def get_anomaly_scores(model, scaler, X):
    X_scaled = scaler.transform(X)
    scores = model.decision_function(X_scaled)
    return scores
