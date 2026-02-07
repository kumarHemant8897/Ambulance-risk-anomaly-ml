import numpy as np

def compute_risk_score(hr_trend, spo2_trend, bp_trend, anomaly_flag):
    hr_weight = 0.3
    spo2_weight = 0.3
    bp_weight = 0.2
    anomaly_weight = 0.2
    
    hr_risk = np.clip(abs(hr_trend) * 10, 0, 100)
    spo2_risk = np.clip(abs(spo2_trend) * 15, 0, 100)
    bp_risk = np.clip(abs(bp_trend) * 8, 0, 100)
    anomaly_risk = 100 if anomaly_flag else 0
    
    risk_score = (hr_risk * hr_weight + 
                  spo2_risk * spo2_weight + 
                  bp_risk * bp_weight + 
                  anomaly_risk * anomaly_weight)
    
    return risk_score

def compute_confidence(hr_trend, spo2_trend, bp_trend, anomaly_flag, data_quality=1.0):
    base_confidence = 0.8
    
    trend_penalty = 0.1 * (np.isnan(hr_trend) + np.isnan(spo2_trend) + np.isnan(bp_trend))
    anomaly_penalty = 0.3 if anomaly_flag else 0.0
    quality_penalty = (1.0 - data_quality) * 0.5
    
    confidence = base_confidence - trend_penalty - anomaly_penalty - quality_penalty
    confidence = np.clip(confidence, 0.0, 1.0)
    
    return confidence

def calculate_risk(hr_trend, spo2_trend, bp_trend, anomaly_flag, data_quality=1.0):
    risk_score = compute_risk_score(hr_trend, spo2_trend, bp_trend, anomaly_flag)
    confidence = compute_confidence(hr_trend, spo2_trend, bp_trend, anomaly_flag, data_quality)
    
    return risk_score, confidence
