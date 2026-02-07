import pandas as pd
import numpy as np

def compute_rolling_features(df, column, window_sizes):
    features = {}
    
    for window in window_sizes:
        window_samples = window
        
        mean_col = f'{column}_mean_{window}s'
        std_col = f'{column}_std_{window}s'
        slope_col = f'{column}_slope_{window}s'
        
        clean_data = df[df['artifact_flag'] == False][column]
        
        rolling_mean = clean_data.rolling(window=window_samples, min_periods=1).mean()
        rolling_std = clean_data.rolling(window=window_samples, min_periods=1).std()
        
        def compute_slope(series):
            if len(series) < 2:
                return np.nan
            x = np.arange(len(series))
            valid_mask = ~np.isnan(series.values)
            if np.sum(valid_mask) < 2:
                return np.nan
            x_clean = x[valid_mask]
            y_clean = series.values[valid_mask]
            if len(x_clean) < 2:
                return np.nan
            slope = np.polyfit(x_clean, y_clean, 1)[0]
            return slope
        
        rolling_slope = clean_data.rolling(window=window_samples, min_periods=2).apply(compute_slope, raw=False)
        
        features[mean_col] = rolling_mean
        features[std_col] = rolling_std
        features[slope_col] = rolling_slope
    
    return features

def extract_rolling_features(df, vitals_columns=['hr', 'spo2', 'bp_sys', 'bp_dia']):
    window_sizes = [30, 60]
    
    if 'artifact_flag' not in df.columns:
        df['artifact_flag'] = False
    
    all_features = {}
    
    for col in vitals_columns:
        if col in df.columns:
            col_features = compute_rolling_features(df, col, window_sizes)
            all_features.update(col_features)
    
    feature_df = pd.DataFrame(all_features, index=df.index)
    
    return feature_df
