"""
Risk scoring module for calculating operational risk scores in ambulance services.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

class RiskScorer:
    """Calculates risk scores for ambulance operations"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.risk_model = None
        self.risk_weights = {
            'response_time': 0.3,
            'resource_availability': 0.2,
            'operational_efficiency': 0.2,
            'environmental_factors': 0.15,
            'historical_performance': 0.15
        }
    
    def calculate_response_time_risk(self, df):
        """Calculate risk based on response times"""
        risk_scores = pd.Series(index=df.index, dtype=float)
        
        # Travel time risk
        if 'travel_time_min' in df.columns:
            travel_risk = np.where(df['travel_time_min'] > 15, 1.0,
                                  np.where(df['travel_time_min'] > 10, 
                                         0.5 + (df['travel_time_min'] - 10) / 10,
                                         df['travel_time_min'] / 20))
            risk_scores += travel_risk * self.risk_weights['response_time'] * 0.4
        
        # On-scene time risk
        if 'on_scene_time_min' in df.columns:
            scene_risk = np.where(df['on_scene_time_min'] > 45, 1.0,
                                 np.where(df['on_scene_time_min'] > 30,
                                        0.5 + (df['on_scene_time_min'] - 30) / 30,
                                        df['on_scene_time_min'] / 60))
            risk_scores += scene_risk * self.risk_weights['response_time'] * 0.3
        
        # Total response time risk
        if 'total_response_time' in df.columns:
            total_risk = np.where(df['total_response_time'] > 30, 1.0,
                                np.where(df['total_response_time'] > 20,
                                       0.5 + (df['total_response_time'] - 20) / 20,
                                       df['total_response_time'] / 40))
            risk_scores += total_risk * self.risk_weights['response_time'] * 0.3
        
        return risk_scores
    
    def calculate_resource_availability_risk(self, df):
        """Calculate risk based on resource availability"""
        risk_scores = pd.Series(index=df.index, dtype=float)
        
        # Crew size risk
        if 'crew_size' in df.columns:
            crew_risk = np.where(df['crew_size'] < 2, 1.0,
                               np.where(df['crew_size'] == 2, 0.7, 0.3))
            risk_scores += crew_risk * self.risk_weights['resource_availability'] * 0.4
        
        # Vehicle type risk
        if 'vehicle_type_numeric' in df.columns:
            vehicle_risk = np.where(df['vehicle_type_numeric'] == 1, 0.2,  # BLS
                                  np.where(df['vehicle_type_numeric'] == 2, 0.5, 0.8))  # ALS, Critical Care
            risk_scores += vehicle_risk * self.risk_weights['resource_availability'] * 0.3
        
        # Equipment status risk (if available)
        if 'equipment_status' in df.columns:
            equip_risk = np.where(df['equipment_status'] == 'Fully Equipped', 0.1,
                                np.where(df['equipment_status'] == 'Partial', 0.5, 1.0))
            risk_scores += equip_risk * self.risk_weights['resource_availability'] * 0.3
        
        return risk_scores
    
    def calculate_operational_efficiency_risk(self, df):
        """Calculate risk based on operational efficiency"""
        risk_scores = pd.Series(index=df.index, dtype=float)
        
        # Crew efficiency risk
        if 'crew_efficiency' in df.columns:
            # Normalize crew efficiency (higher is better, so invert for risk)
            efficiency_normalized = self.scaler.fit_transform(df[['crew_efficiency']])
            efficiency_risk = 1 - efficiency_normalized.flatten()
            risk_scores += efficiency_risk * self.risk_weights['operational_efficiency'] * 0.4
        
        # Time ratio risks
        ratio_cols = [col for col in df.columns if 'ratio' in col]
        for col in ratio_cols[:2]:  # Limit to first 2 ratios
            if col in df.columns:
                # Extreme ratios indicate inefficiency
                ratio_risk = np.abs(df[col] - df[col].median()) / (df[col].std() + 1e-6)
                ratio_risk = np.clip(ratio_risk, 0, 1)
                risk_scores += ratio_risk * self.risk_weights['operational_efficiency'] * 0.3
        
        # Rush hour risk
        if 'is_rush_hour' in df.columns:
            rush_risk = df['is_rush_hour'] * 0.8
            risk_scores += rush_risk * self.risk_weights['operational_efficiency'] * 0.3
        
        return risk_scores
    
    def calculate_environmental_risk(self, df):
        """Calculate risk based on environmental factors"""
        risk_scores = pd.Series(index=df.index, dtype=float)
        
        # Weather risk
        if 'weather_severity' in df.columns:
            weather_risk = df['weather_severity'] / 3.0  # Normalize to 0-1
            risk_scores += weather_risk * self.risk_weights['environmental_factors'] * 0.5
        
        # Time of day risk
        if 'hour' in df.columns:
            # Higher risk during late night hours
            night_risk = np.where((df['hour'] >= 23) | (df['hour'] <= 5), 0.8, 0.2)
            risk_scores += night_risk * self.risk_weights['environmental_factors'] * 0.3
        
        # Weekend risk
        if 'is_weekend' in df.columns:
            weekend_risk = df['is_weekend'] * 0.3
            risk_scores += weekend_risk * self.risk_weights['environmental_factors'] * 0.2
        
        return risk_scores
    
    def calculate_historical_performance_risk(self, df):
        """Calculate risk based on historical performance patterns"""
        risk_scores = pd.Series(index=df.index, dtype=float)
        
        # Location-based historical risk
        if 'location_cluster' in df.columns:
            # Calculate historical performance by location
            location_performance = df.groupby('location_cluster')['travel_time_min'].mean()
            overall_avg = df['travel_time_min'].mean()
            
            # Higher risk for locations with historically poor performance
            location_risk = df['location_cluster'].map(location_performance) / overall_avg
            location_risk = np.clip(location_risk - 1, 0, 1)  # Only positive differences increase risk
            risk_scores += location_risk * self.risk_weights['historical_performance'] * 0.4
        
        # Incident type historical risk
        if 'incident_type' in df.columns:
            incident_performance = df.groupby('incident_type')['total_response_time'].mean()
            overall_avg = df['total_response_time'].mean()
            
            incident_risk = df['incident_type'].map(incident_performance) / overall_avg
            incident_risk = np.clip(incident_risk - 1, 0, 1)
            risk_scores += incident_risk * self.risk_weights['historical_performance'] * 0.3
        
        # Severity-based risk
        if 'severity_numeric' in df.columns:
            severity_risk = df['severity_numeric'] / 3.0  # Normalize to 0-1
            risk_scores += severity_risk * self.risk_weights['historical_performance'] * 0.3
        
        return risk_scores
    
    def calculate_composite_risk_score(self, df):
        """Calculate composite risk score combining all factors"""
        risk_components = {}
        
        # Calculate individual risk components
        risk_components['response_time'] = self.calculate_response_time_risk(df)
        risk_components['resource_availability'] = self.calculate_resource_availability_risk(df)
        risk_components['operational_efficiency'] = self.calculate_operational_efficiency_risk(df)
        risk_components['environmental'] = self.calculate_environmental_risk(df)
        risk_components['historical_performance'] = self.calculate_historical_performance_risk(df)
        
        # Combine into composite score
        composite_risk = sum(risk_components.values())
        
        # Ensure score is between 0 and 1
        composite_risk = np.clip(composite_risk, 0, 1)
        
        return composite_risk, risk_components
    
    def categorize_risk_level(self, risk_scores):
        """Categorize risk scores into levels"""
        risk_levels = pd.cut(risk_scores, 
                            bins=[0, 0.3, 0.6, 0.8, 1.0],
                            labels=['Low', 'Medium', 'High', 'Critical'],
                            include_lowest=True)
        return risk_levels
    
    def train_risk_prediction_model(self, df, target_column='composite_risk'):
        """Train a model to predict risk scores"""
        # Prepare features
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in ['incident_id', target_column]]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_column]
        
        # Train model
        self.risk_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.risk_model.fit(X, y)
        
        print(f"Risk prediction model trained with R² score: {self.risk_model.score(X, y):.3f}")
        return self.risk_model
    
    def predict_risk_scores(self, df):
        """Predict risk scores using trained model"""
        if self.risk_model is None:
            raise ValueError("Risk model not trained. Call train_risk_prediction_model() first.")
        
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in ['incident_id', 'composite_risk']]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        predicted_risk = self.risk_model.predict(X)
        
        return np.clip(predicted_risk, 0, 1)
    
    def generate_risk_report(self, df, risk_scores, save_path='../plots/risk_report.txt'):
        """Generate comprehensive risk report"""
        risk_levels = self.categorize_risk_level(risk_scores)
        
        report = []
        report.append("RISK ASSESSMENT REPORT")
        report.append("=" * 50)
        report.append(f"Total Records: {len(df)}")
        report.append(f"Average Risk Score: {risk_scores.mean():.3f}")
        report.append(f"Max Risk Score: {risk_scores.max():.3f}")
        report.append(f"Min Risk Score: {risk_scores.min():.3f}")
        report.append("")
        
        # Risk level distribution
        risk_dist = risk_levels.value_counts()
        report.append("Risk Level Distribution:")
        report.append("-" * 30)
        for level, count in risk_dist.items():
            percentage = (count / len(df)) * 100
            report.append(f"{level}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # High-risk incidents analysis
        high_risk_mask = risk_scores > 0.7
        if high_risk_mask.sum() > 0:
            report.append("High-Risk Incidents Analysis:")
            report.append("-" * 35)
            
            high_risk_df = df[high_risk_mask]
            
            if 'hour' in high_risk_df.columns:
                report.append(f"Average Hour: {high_risk_df['hour'].mean():.1f}")
            
            if 'incident_type' in high_risk_df.columns:
                common_incidents = high_risk_df['incident_type'].value_counts().head(3)
                report.append("Common Incident Types:")
                for incident_type, count in common_incidents.items():
                    report.append(f"  - {incident_type}: {count}")
            
            if 'severity' in high_risk_df.columns:
                severity_dist = high_risk_df['severity'].value_counts()
                report.append("Severity Distribution:")
                for severity, count in severity_dist.items():
                    report.append(f"  - {severity}: {count}")
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Risk report saved to {save_path}")
        return '\n'.join(report)
    
    def save_model(self, filepath):
        """Save trained risk model"""
        if self.risk_model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'risk_model': self.risk_model,
            'risk_weights': self.risk_weights,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, filepath)
        print(f"Risk model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained risk model"""
        model_data = joblib.load(filepath)
        
        self.risk_model = model_data['risk_model']
        self.risk_weights = model_data['risk_weights']
        self.scaler = model_data['scaler']
        
        print(f"Risk model loaded from {filepath}")
        return self

if __name__ == "__main__":
    # Example usage
    scorer = RiskScorer()
    
    try:
        # Load engineered data
        df = pd.read_csv('../data/processed/incidents_engineered.csv')
        
        # Calculate risk scores
        risk_scores, risk_components = scorer.calculate_composite_risk_score(df)
        
        # Add risk scores to dataframe
        df['composite_risk'] = risk_scores
        df['risk_level'] = scorer.categorize_risk_level(risk_scores)
        
        # Train prediction model
        scorer.train_risk_prediction_model(df)
        
        # Generate report
        report = scorer.generate_risk_report(df, risk_scores)
        
        # Save data with risk scores
        df.to_csv('../data/processed/incidents_with_risk.csv', index=False)
        
        # Save model
        scorer.save_model('../models/risk_scorer.pkl')
        
        print("Risk scoring pipeline completed successfully")
        
    except FileNotFoundError:
        print("Processed data not found. Run feature_engineering.py first.")
