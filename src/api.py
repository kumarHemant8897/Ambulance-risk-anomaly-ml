"""
REST API for ambulance ML models and predictions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging

# Import our modules
from anomaly_model import AnomalyDetector
from risk_score import RiskScorer

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
anomaly_detector = None
risk_scorer = None

def load_models():
    """Load pre-trained models"""
    global anomaly_detector, risk_scorer
    
    try:
        # Load anomaly detector
        if os.path.exists('../models/anomaly_detector.pkl'):
            anomaly_detector = AnomalyDetector()
            anomaly_detector.load_model('../models/anomaly_detector.pkl')
            logger.info("Anomaly detector loaded successfully")
        
        # Load risk scorer
        if os.path.exists('../models/risk_scorer.pkl'):
            risk_scorer = RiskScorer()
            risk_scorer.load_model('../models/risk_scorer.pkl')
            logger.info("Risk scorer loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'anomaly_detector': anomaly_detector is not None,
            'risk_scorer': risk_scorer is not None
        }
    })

@app.route('/predict/anomaly', methods=['POST'])
def predict_anomaly():
    """Predict anomalies in ambulance data"""
    if anomaly_detector is None:
        return jsonify({'error': 'Anomaly detector model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid data format'}), 400
        
        # Predict anomalies
        predictions, scores = anomaly_detector.predict(df)
        
        # Format response
        results = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            results.append({
                'index': i,
                'is_anomaly': bool(pred),
                'anomaly_score': float(score),
                'risk_level': 'High' if pred == 1 else 'Normal'
            })
        
        return jsonify({
            'predictions': results,
            'summary': {
                'total_records': len(predictions),
                'anomalies_detected': int(np.sum(predictions)),
                'anomaly_rate': float(np.mean(predictions))
            }
        })
        
    except Exception as e:
        logger.error(f"Error in anomaly prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/risk', methods=['POST'])
def predict_risk():
    """Predict risk scores for ambulance operations"""
    if risk_scorer is None:
        return jsonify({'error': 'Risk scorer model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid data format'}), 400
        
        # Calculate risk scores
        risk_scores, risk_components = risk_scorer.calculate_composite_risk_score(df)
        risk_levels = risk_scorer.categorize_risk_level(risk_scores)
        
        # Format response
        results = []
        for i, (score, level) in enumerate(zip(risk_scores, risk_levels)):
            components = {}
            for comp_name, comp_scores in risk_components.items():
                components[comp_name] = float(comp_scores.iloc[i])
            
            results.append({
                'index': i,
                'risk_score': float(score),
                'risk_level': str(level),
                'risk_components': components
            })
        
        return jsonify({
            'predictions': results,
            'summary': {
                'total_records': len(risk_scores),
                'average_risk_score': float(np.mean(risk_scores)),
                'max_risk_score': float(np.max(risk_scores)),
                'high_risk_count': int(np.sum(risk_scores > 0.7))
            }
        })
        
    except Exception as e:
        logger.error(f"Error in risk prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/comprehensive', methods=['POST'])
def predict_comprehensive():
    """Comprehensive prediction including both anomaly and risk analysis"""
    if anomaly_detector is None or risk_scorer is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid data format'}), 400
        
        # Get anomaly predictions
        anomaly_predictions, anomaly_scores = anomaly_detector.predict(df)
        
        # Get risk scores
        risk_scores, risk_components = risk_scorer.calculate_composite_risk_score(df)
        risk_levels = risk_scorer.categorize_risk_level(risk_scores)
        
        # Combine results
        results = []
        for i in range(len(df)):
            components = {}
            for comp_name, comp_scores in risk_components.items():
                components[comp_name] = float(comp_scores.iloc[i])
            
            # Determine overall alert level
            is_anomaly = bool(anomaly_predictions[i])
            risk_level = str(risk_levels.iloc[i])
            
            if is_anomaly and risk_level in ['High', 'Critical']:
                overall_alert = 'Critical'
            elif is_anomaly or risk_level in ['High', 'Critical']:
                overall_alert = 'High'
            elif risk_level == 'Medium':
                overall_alert = 'Medium'
            else:
                overall_alert = 'Low'
            
            results.append({
                'index': i,
                'anomaly_detection': {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': float(anomaly_scores[i])
                },
                'risk_assessment': {
                    'risk_score': float(risk_scores.iloc[i]),
                    'risk_level': risk_level,
                    'risk_components': components
                },
                'overall_alert_level': overall_alert,
                'requires_immediate_attention': overall_alert in ['Critical', 'High']
            })
        
        return jsonify({
            'predictions': results,
            'summary': {
                'total_records': len(results),
                'anomalies_detected': int(np.sum(anomaly_predictions)),
                'average_risk_score': float(np.mean(risk_scores)),
                'critical_alerts': len([r for r in results if r['overall_alert_level'] == 'Critical']),
                'high_alerts': len([r for r in results if r['overall_alert_level'] == 'High'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error in comprehensive prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Get analytics summary for dashboard"""
    try:
        # Try to load recent data
        data_path = '../data/processed/incidents_with_risk.csv'
        if not os.path.exists(data_path):
            return jsonify({'error': 'Processed data not found'}), 404
        
        df = pd.read_csv(data_path)
        
        # Calculate summary statistics
        summary = {
            'total_incidents': len(df),
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns else None
            },
            'response_times': {
                'average_travel_time': float(df['travel_time_min'].mean()) if 'travel_time_min' in df.columns else None,
                'average_total_response': float(df['total_response_time'].mean()) if 'total_response_time' in df.columns else None
            },
            'risk_distribution': {},
            'incident_types': {},
            'performance_metrics': {}
        }
        
        # Risk distribution
        if 'risk_level' in df.columns:
            risk_dist = df['risk_level'].value_counts().to_dict()
            summary['risk_distribution'] = {k: int(v) for k, v in risk_dist.items()}
        
        # Incident types
        if 'incident_type' in df.columns:
            incident_dist = df['incident_type'].value_counts().head(5).to_dict()
            summary['incident_types'] = {k: int(v) for k, v in incident_dist.items()}
        
        # Performance metrics
        if 'composite_risk' in df.columns:
            summary['performance_metrics'] = {
                'average_risk_score': float(df['composite_risk'].mean()),
                'high_risk_incidents': int((df['composite_risk'] > 0.7).sum()),
                'critical_incidents': int((df['composite_risk'] > 0.9).sum())
            }
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/info', methods=['GET'])
def get_models_info():
    """Get information about loaded models"""
    info = {
        'anomaly_detector': {
            'loaded': anomaly_detector is not None,
            'type': anomaly_detector.model_type if anomaly_detector else None,
            'features': len(anomaly_detector.feature_names) if anomaly_detector and anomaly_detector.feature_names else 0
        },
        'risk_scorer': {
            'loaded': risk_scorer is not None,
            'model_trained': risk_scorer.risk_model is not None if risk_scorer else False,
            'risk_weights': risk_scorer.risk_weights if risk_scorer else None
        }
    }
    
    return jsonify(info)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
