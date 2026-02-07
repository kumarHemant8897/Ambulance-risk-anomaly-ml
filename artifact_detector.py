import pandas as pd
import numpy as np

def detect_hr_artifacts(df, hr_col='hr', window=5, threshold=20):
    hr_diff = df[hr_col].diff().abs()
    hr_rolling = hr_diff.rolling(window=window, center=True).max()
    hr_artifacts = hr_rolling > threshold
    return hr_artifacts.fillna(False)

def detect_spo2_artifacts(df, spo2_col='spo2', drop_threshold=5, window=3):
    spo2_diff = df[spo2_col].diff()
    spo2_drops = spo2_diff < -drop_threshold
    spo2_rolling = spo2_drops.rolling(window=window, center=True).apply(lambda x: any(x), raw=True)
    return spo2_rolling.fillna(False)

def detect_motion_artifacts(df, motion_col='motion', threshold=2.0):
    motion_artifacts = df[motion_col] > threshold
    return motion_artifacts.fillna(False)

def detect_signal_artifacts(df):
    df_copy = df.copy()
    
    hr_artifacts = detect_hr_artifacts(df_copy)
    spo2_artifacts = detect_spo2_artifacts(df_copy)
    motion_artifacts = detect_motion_artifacts(df_copy)
    
    df_copy['artifact_flag'] = hr_artifacts | spo2_artifacts | motion_artifacts
    
    return df_copy
