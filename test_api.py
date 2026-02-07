import requests
import json

# Test data with 5 samples
test_data = {
    "timestamp": [
        "2026-02-07T20:53:11",
        "2026-02-07T20:53:12", 
        "2026-02-07T20:53:13",
        "2026-02-07T20:53:14",
        "2026-02-07T20:53:15"
    ],
    "hr": [75, 76, 74, 78, 77],
    "spo2": [98, 97, 98, 99, 98],
    "bp_sys": [120, 122, 118, 125, 121],
    "bp_dia": [80, 82, 78, 85, 81],
    "motion": [0.5, 0.3, 0.4, 0.6, 0.2]
}

response = requests.post("http://localhost:8000/predict", json=test_data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
