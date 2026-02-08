Here is a **short, clean, internship-ready reduced README** with only the most important information kept.

---

# Gray Ambulance ML Project

An end-to-end **machine learning system** for analyzing ambulance operations, detecting anomalies, and generating operational **risk scores**, with REST API deployment.

**Author:** Hament Kumar
**Assignment:** Gray Mobility Internship Project

---

## Project Overview

This project simulates real ambulance incident data and builds a complete ML pipeline that includes:

* **Synthetic data generation** with realistic medical, temporal, and geographic patterns
* **Data quality & anomaly detection** using Isolation Forest and One-Class SVM
* **Feature engineering** with temporal, location, and performance indicators
* **Composite risk scoring** based on response time, resources, environment, and history
* **Flask REST API** for real-time anomaly and risk prediction

---

## Tech Stack

* Python, Pandas, NumPy, Scikit-learn
* Isolation Forest, One-Class SVM, Random Forest
* Feature Engineering & Anomaly Detection
* Flask REST API
* Matplotlib / Seaborn for visualization

---

## Project Structure

```
data/        # Raw and processed datasets  
src/         # ML pipeline and API code  
models/      # Trained models  
plots/       # Visualizations  
requirements.txt  
README.md  
```

---

## How to Run

```bash
pip install -r requirements.txt

python src/generate_data.py
python src/artifact_detection.py
python src/feature_engineering.py
python src/anomaly_model.py
python src/risk_score.py
python src/api.py
```

API runs at: **[http://localhost:5000](http://localhost:5000)**

---

## Key Results

* ~90% anomaly detection precision on synthetic incidents
* Risk score strongly correlated with incident severity (**R² ≈ 0.78**)
* Real-time prediction via REST API

---

## License

MIT License.

---

If you'd like, I can also create a **perfect GitHub description + resume project description** to increase your internship selection chances.
