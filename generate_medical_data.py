import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_medical_timeseries():
    sampling_rate = 1
    duration_minutes = 30
    total_samples = duration_minutes * 60 * sampling_rate
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i) for i in range(total_samples)]
    
    time = np.arange(total_samples)
    
    hr_base = 75
    hr_trend = 0.001 * time
    hr_noise = np.random.normal(0, 3, total_samples)
    hr = hr_base + hr_trend + hr_noise + 5 * np.sin(2 * np.pi * time / 300)
    
    spo2_base = 98
    spo2_trend = -0.0005 * time
    spo2_noise = np.random.normal(0, 0.5, total_samples)
    spo2 = spo2_base + spo2_trend + spo2_noise + 2 * np.sin(2 * np.pi * time / 200)
    spo2 = np.clip(spo2, 85, 100)
    
    bp_sys_base = 120
    bp_sys_trend = 0.002 * time
    bp_sys_noise = np.random.normal(0, 4, total_samples)
    bp_sys = bp_sys_base + bp_sys_trend + bp_sys_noise + 8 * np.sin(2 * np.pi * time / 400)
    
    bp_dia_base = 80
    bp_dia_trend = 0.001 * time
    bp_dia_noise = np.random.normal(0, 3, total_samples)
    bp_dia = bp_dia_base + bp_dia_trend + bp_dia_noise + 5 * np.sin(2 * np.pi * time / 400)
    
    motion_base = 0.1
    motion_noise = np.random.exponential(0.2, total_samples)
    motion = motion_base + motion_noise + 0.5 * np.sin(2 * np.pi * time / 60)
    motion = np.clip(motion, 0, 5)
    
    missing_segments = [(300, 320), (900, 915), (1200, 1225), (1500, 1520)]
    
    for start, end in missing_segments:
        hr[start:end] = np.nan
        spo2[start:end] = np.nan
        bp_sys[start:end] = np.nan
        bp_dia[start:end] = np.nan
        motion[start:end] = np.nan
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'hr': hr,
        'spo2': spo2,
        'bp_sys': bp_sys,
        'bp_dia': bp_dia,
        'motion': motion
    })
    
    return data

if __name__ == "__main__":
    data = generate_medical_timeseries()
    data.to_csv('medical_timeseries.csv', index=False)
    print(f"Generated {len(data)} samples")
