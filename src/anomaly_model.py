"""
Anomaly detection model for identifying unusual patterns in ambulance operations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyDetector:
    """Detects anomalies in ambulance operational data"""
    
    def __init__(self, model_type='isolation_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Initialize model based on type
        if model_type == 'isolation_forest':
            self.model = IsolationForest(contamination=0.1, random_state=42)
        elif model_type == 'one_class_svm':
            self.model = OneClassSVM(nu=0.1, kernel='rbf')
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def preprocess_data(self, df, exclude_columns=None):
        """Preprocess data for anomaly detection"""
        if exclude_columns is None:
            exclude_columns = ['incident_id', 'vehicle_id', 'timestamp']
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        # Handle missing values
        X = df[feature_cols].fillna(df[feature_cols].median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = feature_cols
        return X_scaled, feature_cols
    
    def fit(self, df, exclude_columns=None):
        """Train the anomaly detection model"""
        X_scaled, feature_cols = self.preprocess_data(df, exclude_columns)
        
        print(f"Training {self.model_type} model on {len(feature_cols)} features...")
        self.model.fit(X_scaled)
        
        print("Model training completed")
        return self
    
    def predict(self, df):
        """Predict anomalies in new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled, _ = self.preprocess_data(df)
        
        if self.model_type in ['isolation_forest', 'one_class_svm']:
            # These models return -1 for anomalies, 1 for normal
            predictions = self.model.predict(X_scaled)
            anomaly_scores = self.model.decision_function(X_scaled)
            
            # Convert to binary: 1 for anomaly, 0 for normal
            binary_predictions = (predictions == -1).astype(int)
            
            return binary_predictions, anomaly_scores
        else:
            # For supervised models
            predictions = self.model.predict(X_scaled)
            if hasattr(self.model, 'predict_proba'):
                anomaly_scores = self.model.predict_proba(X_scaled)[:, 1]
            else:
                anomaly_scores = predictions
                
            return predictions, anomaly_scores
    
    def evaluate(self, df, true_labels=None):
        """Evaluate model performance"""
        predictions, scores = self.predict(df)
        
        if true_labels is not None:
            # Supervised evaluation
            print("Classification Report:")
            print(classification_report(true_labels, predictions))
            
            print("Confusion Matrix:")
            cm = confusion_matrix(true_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('../plots/confusion_matrix.png')
            plt.close()
            
            if len(np.unique(true_labels)) == 2:
                auc_score = roc_auc_score(true_labels, scores)
                print(f"ROC AUC Score: {auc_score:.3f}")
        
        # Unsupervised evaluation metrics
        anomaly_rate = np.mean(predictions)
        print(f"Anomaly Detection Rate: {anomaly_rate:.3f}")
        print(f"Number of anomalies detected: {np.sum(predictions)}")
        
        return predictions, scores
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance for tree-based models"""
        if self.model_type == 'random_forest' and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance.head(top_n)
        else:
            print("Feature importance not available for this model type")
            return None
    
    def plot_anomaly_scores(self, scores, threshold=None, save_path='../plots/anomaly_scores.png'):
        """Plot distribution of anomaly scores"""
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores')
        
        if threshold is not None:
            plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_anomalies_over_time(self, df, scores, timestamp_col='timestamp', save_path='../plots/anomalies_timeline.png'):
        """Plot anomalies over time"""
        if timestamp_col not in df.columns:
            print(f"Timestamp column '{timestamp_col}' not found")
            return
        
        df_temp = df.copy()
        df_temp['anomaly_score'] = scores
        df_temp['timestamp'] = pd.to_datetime(df_temp[timestamp_col])
        
        # Sort by timestamp
        df_temp = df_temp.sort_values('timestamp')
        
        plt.figure(figsize=(15, 6))
        plt.plot(df_temp['timestamp'], df_temp['anomaly_score'], alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Scores Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filepath}")
        return self

class AnomalyAnalyzer:
    """Analyze and interpret detected anomalies"""
    
    def __init__(self):
        pass
    
    def analyze_anomaly_patterns(self, df, predictions):
        """Analyze patterns in detected anomalies"""
        df_analysis = df.copy()
        df_analysis['is_anomaly'] = predictions
        
        # Summary statistics
        anomaly_summary = {
            'total_records': len(df_analysis),
            'anomaly_count': df_analysis['is_anomaly'].sum(),
            'anomaly_rate': df_analysis['is_anomaly'].mean()
        }
        
        # Anomaly patterns by different dimensions
        patterns = {}
        
        # By hour of day
        if 'hour' in df_analysis.columns:
            patterns['by_hour'] = df_analysis.groupby('hour')['is_anomaly'].agg(['count', 'sum', 'mean'])
        
        # By day of week
        if 'day_of_week' in df_analysis.columns:
            patterns['by_day_of_week'] = df_analysis.groupby('day_of_week')['is_anomaly'].agg(['count', 'sum', 'mean'])
        
        # By incident type
        if 'incident_type' in df_analysis.columns:
            patterns['by_incident_type'] = df_analysis.groupby('incident_type')['is_anomaly'].agg(['count', 'sum', 'mean'])
        
        # By severity
        if 'severity' in df_analysis.columns:
            patterns['by_severity'] = df_analysis.groupby('severity')['is_anomaly'].agg(['count', 'sum', 'mean'])
        
        return anomaly_summary, patterns
    
    def generate_anomaly_report(self, df, predictions, save_path='../plots/anomaly_report.txt'):
        """Generate comprehensive anomaly report"""
        anomaly_summary, patterns = self.analyze_anomaly_patterns(df, predictions)
        
        report = []
        report.append("ANOMALY DETECTION REPORT")
        report.append("=" * 50)
        report.append(f"Total Records: {anomaly_summary['total_records']}")
        report.append(f"Anomalies Detected: {anomaly_summary['anomaly_count']}")
        report.append(f"Anomaly Rate: {anomaly_summary['anomaly_rate']:.3f}")
        report.append("")
        
        for pattern_name, pattern_data in patterns.items():
            report.append(f"Anomaly Patterns by {pattern_name.replace('by_', '').replace('_', ' ').title()}:")
            report.append("-" * 30)
            report.append(pattern_data.to_string())
            report.append("")
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Anomaly report saved to {save_path}")
        return '\n'.join(report)

if __name__ == "__main__":
    # Example usage
    detector = AnomalyDetector(model_type='isolation_forest')
    
    try:
        # Load engineered data
        df = pd.read_csv('../data/processed/incidents_engineered.csv')
        
        # Train model
        detector.fit(df)
        
        # Predict anomalies
        predictions, scores = detector.predict(df)
        
        # Evaluate
        detector.evaluate(df)
        
        # Analyze anomalies
        analyzer = AnomalyAnalyzer()
        report = analyzer.generate_anomaly_report(df, predictions)
        
        # Save model
        detector.save_model('../models/anomaly_detector.pkl')
        
        print("Anomaly detection pipeline completed successfully")
        
    except FileNotFoundError:
        print("Processed data not found. Run feature_engineering.py first.")
