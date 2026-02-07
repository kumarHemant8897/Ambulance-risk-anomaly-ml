"""
Artifact detection module for identifying data quality issues and anomalies.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ArtifactDetector:
    """Detects artifacts and data quality issues in ambulance data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
    def detect_missing_data(self, df):
        """Detect missing data patterns"""
        missing_info = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum(),
            'missing_percentage': (df.isnull().sum() / len(df)) * 100
        }
        return missing_info
    
    def detect_outliers(self, df, numeric_columns):
        """Detect outliers using IQR method"""
        outliers_info = {}
        
        for col in numeric_columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
                
        return outliers_info
    
    def detect_temporal_anomalies(self, df, timestamp_col, value_col):
        """Detect temporal anomalies in time series data"""
        if timestamp_col not in df.columns or value_col not in df.columns:
            return {}
            
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col)
        
        # Calculate rolling statistics
        window_size = min(24, len(df_sorted) // 10)  # Adaptive window size
        rolling_mean = df_sorted[value_col].rolling(window=window_size).mean()
        rolling_std = df_sorted[value_col].rolling(window=window_size).std()
        
        # Identify anomalies (values > 3 std from rolling mean)
        anomalies = df_sorted[
            np.abs(df_sorted[value_col] - rolling_mean) > 3 * rolling_std
        ]
        
        return {
            'anomaly_count': len(anomalies),
            'anomaly_percentage': (len(anomalies) / len(df)) * 100,
            'anomaly_indices': anomalies.index.tolist()
        }
    
    def detect_geographic_anomalies(self, df, lat_col, lon_col):
        """Detect geographic anomalies in location data"""
        if lat_col not in df.columns or lon_col not in df.columns:
            return {}
            
        # Calculate distance from mean location
        mean_lat = df[lat_col].mean()
        mean_lon = df[lon_col].mean()
        
        # Simple distance calculation (approximate)
        distances = np.sqrt((df[lat_col] - mean_lat)**2 + (df[lon_col] - mean_lon)**2)
        
        # Use Isolation Forest for anomaly detection
        location_data = df[[lat_col, lon_col]].dropna()
        if len(location_data) > 0:
            anomalies = self.isolation_forest.fit_predict(location_data)
            anomaly_indices = location_data[anomalies == -1].index.tolist()
            
            return {
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(df)) * 100,
                'anomaly_indices': anomaly_indices,
                'mean_location': (mean_lat, mean_lon)
            }
        
        return {}
    
    def detect_response_time_anomalies(self, df):
        """Detect anomalies in response time patterns"""
        response_cols = ['dispatch_time_min', 'travel_time_min', 'on_scene_time_min', 'transport_time_min']
        anomalies = {}
        
        for col in response_cols:
            if col in df.columns:
                # Check for negative values
                negative_count = (df[col] < 0).sum()
                
                # Check for extremely high values (> 99th percentile)
                high_threshold = df[col].quantile(0.99)
                high_count = (df[col] > high_threshold).sum()
                
                anomalies[col] = {
                    'negative_values': negative_count,
                    'extremely_high_values': high_count,
                    'high_threshold': high_threshold
                }
                
        return anomalies
    
    def generate_quality_report(self, df, save_path=None):
        """Generate comprehensive data quality report"""
        report = {
            'dataset_shape': df.shape,
            'missing_data': self.detect_missing_data(df),
            'numeric_outliers': self.detect_outliers(df, df.select_dtypes(include=[np.number]).columns),
            'response_time_anomalies': self.detect_response_time_anomalies(df)
        }
        
        # Add temporal anomalies if timestamp column exists
        timestamp_cols = ['timestamp', 'date', 'time']
        for col in timestamp_cols:
            if col in df.columns:
                value_cols = df.select_dtypes(include=[np.number]).columns
                if len(value_cols) > 0:
                    report['temporal_anomalies'] = self.detect_temporal_anomalies(df, col, value_cols[0])
                break
        
        # Add geographic anomalies if location columns exist
        if 'latitude' in df.columns and 'longitude' in df.columns:
            report['geographic_anomalies'] = self.detect_geographic_anomalies(df, 'latitude', 'longitude')
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        return report
    
    def plot_data_quality(self, df, save_dir='../plots'):
        """Create data quality visualization plots"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Missing data heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
        plt.title('Missing Data Pattern')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/missing_data_heatmap.png')
        plt.close()
        
        # Numeric distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6 plots
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols):
                if i < 6:
                    axes[i].hist(df[col].dropna(), bins=30, alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    
            plt.tight_layout()
            plt.savefig(f'{save_dir}/numeric_distributions.png')
            plt.close()

if __name__ == "__main__":
    # Example usage
    detector = ArtifactDetector()
    
    # Load sample data
    try:
        df = pd.read_csv('../data/raw/incidents.csv')
        report = detector.generate_quality_report(df, '../plots/data_quality_report.json')
        detector.plot_data_quality(df)
        print("Data quality analysis completed")
    except FileNotFoundError:
        print("Sample data not found. Run generate_data.py first.")
