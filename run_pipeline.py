import pandas as pd
from artifact_detector import detect_signal_artifacts
from feature_extraction import extract_rolling_features
from isolation_forest_anomaly import train_model, predict
from risk_scoring import calculate_risk

# Load data
df = pd.read_csv('medical_timeseries.csv')

# Detect artifacts
df_with_artifacts = detect_signal_artifacts(df)
print(f'Artifacts detected: {df_with_artifacts["artifact_flag"].sum()}')

# Extract features
features_df = extract_rolling_features(df_with_artifacts)
print(f'Features extracted: {features_df.shape}')

# Train model
model, scaler = train_model(features_df.dropna())
print('Model trained')

# Make predictions
predictions = predict(model, scaler, features_df.fillna(features_df.mean()))
print(f'Anomalies predicted: {(predictions == -1).sum()}')

# Calculate risk for last sample
hr_trend = features_df['hr_slope_30s'].iloc[-1]
spo2_trend = features_df['spo2_slope_30s'].iloc[-1]
bp_trend = features_df['bp_sys_slope_30s'].iloc[-1]
artifact_flag = df_with_artifacts['artifact_flag'].iloc[-1]
risk_score, confidence = calculate_risk(hr_trend, spo2_trend, bp_trend, artifact_flag)

print(f'Final risk score: {risk_score:.2f}, confidence: {confidence:.2f}')
