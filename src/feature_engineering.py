"""
Feature engineering module for creating ML-ready features from ambulance data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import holidays

class FeatureEngineer:
    """Engineers features for ambulance ML models"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.us_holidays = holidays.US()
        
    def create_temporal_features(self, df, timestamp_col='timestamp'):
        """Create time-based features"""
        if timestamp_col not in df.columns:
            return df
            
        df = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract time components
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['quarter'] = df[timestamp_col].dt.quarter
        df['year'] = df[timestamp_col].dt.year
        
        # Create cyclical features for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Rush hour indicator
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(16, 19))).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Holiday indicator
        df['is_holiday'] = df[timestamp_col].dt.date.apply(lambda x: x in self.us_holidays).astype(int)
        
        return df
    
    def create_response_time_features(self, df):
        """Create features related to response times"""
        df = df.copy()
        
        # Total response time
        if 'dispatch_time_min' in df.columns and 'travel_time_min' in df.columns:
            df['total_response_time'] = df['dispatch_time_min'] + df['travel_time_min']
        
        # Time ratios
        time_cols = ['dispatch_time_min', 'travel_time_min', 'on_scene_time_min', 'transport_time_min']
        available_cols = [col for col in time_cols if col in df.columns]
        
        if len(available_cols) >= 2:
            for i, col1 in enumerate(available_cols):
                for col2 in available_cols[i+1:]:
                    ratio_name = f'{col1}_to_{col2}_ratio'
                    df[ratio_name] = df[col1] / (df[col2] + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Performance indicators
        if 'travel_time_min' in df.columns:
            df['is_fast_response'] = (df['travel_time_min'] <= 8).astype(int)  # 8 minutes threshold
            df['is_slow_response'] = (df['travel_time_min'] >= 15).astype(int)  # 15 minutes threshold
        
        return df
    
    def create_location_features(self, df):
        """Create location-based features"""
        df = df.copy()
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Distance from city center (assuming NYC coordinates)
            nyc_lat, nyc_lon = 40.7128, -74.0060
            df['distance_from_center'] = np.sqrt(
                (df['latitude'] - nyc_lat)**2 + (df['longitude'] - nyc_lon)**2
            )
            
            # Create location clusters (simplified grid)
            df['lat_grid'] = (df['latitude'] * 100).astype(int)
            df['lon_grid'] = (df['longitude'] * 100).astype(int)
            df['location_cluster'] = df['lat_grid'].astype(str) + '_' + df['lon_grid'].astype(str)
        
        return df
    
    def create_incident_features(self, df):
        """Create incident-specific features"""
        df = df.copy()
        
        # Severity encoding
        if 'severity' in df.columns:
            severity_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
            df['severity_numeric'] = df['severity'].map(severity_mapping)
        
        # Crew size efficiency
        if 'crew_size' in df.columns and 'on_scene_time_min' in df.columns:
            df['crew_efficiency'] = df['crew_size'] / (df['on_scene_time_min'] + 1e-6)
        
        # Vehicle type encoding
        if 'vehicle_type' in df.columns:
            vehicle_mapping = {'BLS': 1, 'ALS': 2, 'Critical Care': 3}
            df['vehicle_type_numeric'] = df['vehicle_type'].map(vehicle_mapping)
        
        return df
    
    def create_weather_features(self, df):
        """Create weather-related features"""
        df = df.copy()
        
        if 'weather_condition' in df.columns:
            # Weather severity encoding
            weather_severity = {
                'Clear': 0,
                'Fog': 1,
                'Rain': 2,
                'Snow': 3
            }
            df['weather_severity'] = df['weather_condition'].map(weather_severity)
            
            # Adverse weather indicator
            df['is_adverse_weather'] = (df['weather_condition'] != 'Clear').astype(int)
        
        return df
    
    def encode_categorical_features(self, df, categorical_columns=None):
        """Encode categorical features using label encoding"""
        df = df.copy()
        
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Handle unseen categories
                df[col] = df[col].astype(str)
                df[col] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between variables"""
        df = df.copy()
        
        # Time-weather interactions
        if 'hour' in df.columns and 'weather_severity' in df.columns:
            df['hour_weather_interaction'] = df['hour'] * df['weather_severity']
        
        # Location-severity interactions
        if 'distance_from_center' in df.columns and 'severity_numeric' in df.columns:
            df['distance_severity_interaction'] = df['distance_from_center'] * df['severity_numeric']
        
        # Crew-vehicle interactions
        if 'crew_size' in df.columns and 'vehicle_type_numeric' in df.columns:
            df['crew_vehicle_interaction'] = df['crew_size'] * df['vehicle_type_numeric']
        
        return df
    
    def scale_features(self, df, numeric_columns=None):
        """Scale numeric features"""
        df = df.copy()
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns and target variables from scaling
        exclude_cols = ['incident_id', 'vehicle_id']
        numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
        
        if numeric_columns:
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        
        return df
    
    def engineer_features(self, df, target_column=None):
        """Complete feature engineering pipeline"""
        print(f"Starting feature engineering for dataset with shape: {df.shape}")
        
        # Apply all feature engineering steps
        df = self.create_temporal_features(df)
        df = self.create_response_time_features(df)
        df = self.create_location_features(df)
        df = self.create_incident_features(df)
        df = self.create_weather_features(df)
        df = self.create_interaction_features(df)
        
        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            df = self.encode_categorical_features(df, categorical_cols)
        
        print(f"Feature engineering completed. Final shape: {df.shape}")
        print(f"Created features: {list(df.columns)}")
        
        return df
    
    def prepare_ml_data(self, df, target_column, test_size=0.2, random_state=42):
        """Prepare data for ML modeling"""
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Remove ID columns
        id_cols = ['incident_id', 'vehicle_id', 'timestamp']
        X = X.drop(columns=[col for col in id_cols if col in X.columns])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None
        )
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    
    try:
        # Load and process data
        df = pd.read_csv('../data/raw/incidents.csv')
        
        # Engineer features
        df_engineered = engineer.engineer_features(df)
        
        # Save processed data
        df_engineered.to_csv('../data/processed/incidents_engineered.csv', index=False)
        print("Feature engineering completed and data saved")
        
    except FileNotFoundError:
        print("Raw data not found. Run generate_data.py first.")
