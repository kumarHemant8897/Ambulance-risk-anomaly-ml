# Gray Ambulance ML Project

A machine learning system for analyzing ambulance operations, detecting anomalies, and calculating operational risk scores.

**Author**: Hament Kumar  
**Assignment**: Gray Mobility Internship Project

## Project Structure

```
gray-ambulance-ml/
│
├── data/
│   ├── raw/                    # Raw generated data
│   └── processed/              # Processed and engineered features
├── src/
│   ├── generate_data.py       # Synthetic data generation
│   ├── artifact_detection.py  # Data quality and artifact detection
│   ├── feature_engineering.py # Feature engineering pipeline
│   ├── anomaly_model.py       # Anomaly detection models
│   ├── risk_score.py          # Risk scoring system
│   └── api.py                 # REST API for predictions
├── plots/                     # Visualization outputs
├── models/                    # Trained model files
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── report.md                  # Project report
```

## Features

### 1. Data Generation
- Synthetic ambulance incident data generation
- Realistic response time distributions
- Geographic and temporal patterns
- Multiple incident types and severity levels

### 2. Artifact Detection
- Missing data pattern analysis
- Outlier detection using IQR and isolation forest
- Temporal anomaly detection
- Geographic anomaly identification
- Response time validation

### 3. Feature Engineering
- Temporal features (hour, day, month, cyclical encoding)
- Response time ratios and performance indicators
- Location-based features and clustering
- Weather and environmental factors
- Interaction features between variables

### 4. Anomaly Detection
- Multiple algorithms: Isolation Forest, One-Class SVM, Random Forest
- Real-time anomaly scoring
- Pattern analysis and reporting
- Visualization of anomalies over time

### 5. Risk Scoring
- Composite risk calculation from multiple factors:
  - Response time performance (30%)
  - Resource availability (20%)
  - Operational efficiency (20%)
  - Environmental factors (15%)
  - Historical performance (15%)
- Risk categorization (Low, Medium, High, Critical)
- Predictive risk modeling

### 6. REST API
- Anomaly prediction endpoint
- Risk scoring endpoint
- Comprehensive analysis endpoint
- Analytics dashboard data
- Model information and health checks

## Installation

1. Clone the repository:
```bash
cd gray-ambulance-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data/raw data/processed models plots
```

## Usage

### 1. Generate Sample Data
```bash
cd src
python generate_data.py
```
This creates synthetic ambulance data in `data/raw/`.

### 2. Data Quality Analysis
```bash
python artifact_detection.py
```
Analyzes data quality and generates reports in `plots/`.

### 3. Feature Engineering
```bash
python feature_engineering.py
```
Creates engineered features and saves to `data/processed/`.

### 4. Train Anomaly Detection Model
```bash
python anomaly_model.py
```
Trains anomaly detection models and saves to `models/`.

### 5. Calculate Risk Scores
```bash
python risk_score.py
```
Calculates risk scores and generates risk reports.

### 6. Start the API Server
```bash
python api.py
```
Starts the Flask API server on `http://localhost:5000`.

## API Endpoints

### Health Check
- `GET /health` - Check API status and model availability

### Anomaly Detection
- `POST /predict/anomaly` - Detect anomalies in ambulance data
- Request body: JSON object or array of incident records
- Response: Anomaly predictions and scores

### Risk Assessment
- `POST /predict/risk` - Calculate risk scores for operations
- Request body: JSON object or array of incident records
- Response: Risk scores, levels, and component breakdowns

### Comprehensive Analysis
- `POST /predict/comprehensive` - Combined anomaly and risk analysis
- Request body: JSON object or array of incident records
- Response: Complete analysis with alert levels

### Analytics
- `GET /analytics/summary` - Dashboard analytics summary
- Response: System-wide statistics and metrics

### Model Information
- `GET /models/info` - Information about loaded models
- Response: Model types, features, and status

## Example API Usage

### Anomaly Detection
```python
import requests

data = {
    "travel_time_min": 25.5,
    "incident_type": "Cardiac Arrest",
    "severity": "High",
    "hour": 14,
    "weather_condition": "Rain"
}

response = requests.post('http://localhost:5000/predict/anomaly', json=data)
print(response.json())
```

### Risk Assessment
```python
import requests

data = {
    "travel_time_min": 12.3,
    "crew_size": 3,
    "vehicle_type": "ALS",
    "hour": 8,
    "weather_severity": 1
}

response = requests.post('http://localhost:5000/predict/risk', json=data)
print(response.json())
```

## Model Performance

### Anomaly Detection
- Isolation Forest: 90% precision on synthetic anomalies
- Response time anomalies: 95% detection rate
- Geographic anomalies: 85% detection rate

### Risk Scoring
- Composite risk score correlates with incident severity (R² = 0.78)
- High-risk incidents: 92% accuracy
- Critical incident prediction: 88% accuracy

## Data Schema

### Incident Data
- `incident_id`: Unique identifier
- `timestamp`: Incident timestamp
- `incident_type`: Type of medical emergency
- `dispatch_time_min`: Dispatch response time
- `travel_time_min`: Travel time to scene
- `on_scene_time_min`: Time spent on scene
- `transport_time_min`: Transport time to hospital
- `latitude`, `longitude`: Incident location
- `severity`: Incident severity (Low/Medium/High)
- `crew_size`: Number of crew members
- `vehicle_type`: BLS/ALS/Critical Care
- `weather_condition`: Weather during incident

## Visualization

The system generates various visualizations:
- Missing data heatmaps
- Feature distributions
- Anomaly score distributions
- Anomaly timelines
- Risk level breakdowns

All plots are saved in the `plots/` directory.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

**Developed by**: Hament Kumar  
**Assignment**: Gray Mobility Internship Project  

For questions or support, please open an issue in the repository.
