"""
Data generation module for Gray Ambulance ML project.
Generates synthetic ambulance operational data for ML model development.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class AmbulanceDataGenerator:
    """Generates synthetic ambulance operational data"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_incident_data(self, num_records=10000):
        """Generate incident response data"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        data = []
        for i in range(num_records):
            incident_time = start_date + timedelta(
                seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            # Generate incident types
            incident_types = ['Medical Emergency', 'Trauma', 'Cardiac Arrest', 'Stroke', 'Respiratory']
            incident_type = np.random.choice(incident_types)
            
            # Generate response times (in minutes)
            dispatch_time = np.random.uniform(1, 5)
            travel_time = np.random.exponential(8) + 2  # Average 10 minutes
            on_scene_time = np.random.exponential(15) + 5  # Average 20 minutes
            transport_time = np.random.exponential(12) + 3  # Average 15 minutes
            
            # Generate locations (simplified)
            lat = 40.7128 + np.random.uniform(-0.1, 0.1)
            lon = -74.0060 + np.random.uniform(-0.1, 0.1)
            
            # Generate patient outcomes
            severity = np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1])
            
            record = {
                'incident_id': f'INC_{i:06d}',
                'timestamp': incident_time,
                'incident_type': incident_type,
                'dispatch_time_min': dispatch_time,
                'travel_time_min': travel_time,
                'on_scene_time_min': on_scene_time,
                'transport_time_min': transport_time,
                'total_response_time_min': dispatch_time + travel_time,
                'latitude': lat,
                'longitude': lon,
                'severity': severity,
                'crew_size': np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2]),
                'vehicle_type': np.random.choice(['BLS', 'ALS', 'Critical Care'], p=[0.4, 0.5, 0.1]),
                'day_of_week': incident_time.weekday(),
                'hour_of_day': incident_time.hour,
                'weather_condition': np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], p=[0.6, 0.2, 0.1, 0.1])
            }
            data.append(record)
            
        return pd.DataFrame(data)
    
    def generate_vehicle_data(self, num_vehicles=50):
        """Generate vehicle maintenance and status data"""
        data = []
        vehicle_types = ['BLS', 'ALS', 'Critical Care']
        
        for i in range(num_vehicles):
            vehicle_id = f'VEH_{i:03d}'
            vehicle_type = np.random.choice(vehicle_types)
            
            # Generate maintenance records
            last_maintenance = datetime.now() - timedelta(days=np.random.randint(1, 180))
            mileage = np.random.uniform(10000, 150000)
            
            record = {
                'vehicle_id': vehicle_id,
                'vehicle_type': vehicle_type,
                'last_maintenance_date': last_maintenance,
                'mileage': mileage,
                'status': np.random.choice(['Active', 'Maintenance', 'Out of Service'], p=[0.8, 0.15, 0.05]),
                'age_years': np.random.uniform(1, 10),
                'fuel_level': np.random.uniform(0.2, 1.0),
                'equipment_status': np.random.choice(['Fully Equipped', 'Partial', 'Missing Items'], p=[0.85, 0.12, 0.03])
            }
            data.append(record)
            
        return pd.DataFrame(data)
    
    def save_data(self, incident_data, vehicle_data, output_dir='../data/raw'):
        """Save generated data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        incident_data.to_csv(f'{output_dir}/incidents.csv', index=False)
        vehicle_data.to_csv(f'{output_dir}/vehicles.csv', index=False)
        print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    generator = AmbulanceDataGenerator()
    
    # Generate data
    incidents = generator.generate_incident_data(10000)
    vehicles = generator.generate_vehicle_data(50)
    
    # Save data
    generator.save_data(incidents, vehicles)
    
    print(f"Generated {len(incidents)} incident records and {len(vehicles)} vehicle records")
