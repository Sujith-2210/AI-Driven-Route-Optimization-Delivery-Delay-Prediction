"""
Real-Time Prediction Pipeline
Handles new delivery predictions, route optimization, and logging.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Dict, Optional, List
from route_optimizer import optimize_deliveries, RouteOptimizer, DeliveryNode


class DeliveryPredictor:
    """
    Real-time delivery delay prediction and route optimization.
    """
    
    def __init__(self, model_path: str = 'models/best_delay_prediction_model.pkl',
                 encoders_path: str = 'models/label_encoders.pkl',
                 log_path: str = 'logs/predictions.csv'):
        """
        Initialize the predictor with trained model and encoders.
        """
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoders_path)
        self.log_path = log_path
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Feature columns expected by model
        self.feature_cols = ['distance_km', 'package_weight_kg', 'vehicle_type',
                            'traffic_level', 'weather_condition', 'road_type']
        
        print(f"Model loaded: {type(self.model).__name__}")
    
    def preprocess(self, data: Dict) -> pd.DataFrame:
        """
        Preprocess input data for model prediction.
        
        Args:
            data: Dict with delivery features
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        for col in ['vehicle_type', 'traffic_level', 'weather_condition', 'road_type']:
            if col in df.columns and col in self.encoders:
                # Handle unknown categories
                known_classes = set(self.encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else self.encoders[col].classes_[0])
                df[col] = self.encoders[col].transform(df[col])
        
        return df[self.feature_cols]
    
    def predict_delay(self, delivery_data: Dict) -> Dict:
        """
        Predict if a delivery will be delayed.
        
        Args:
            delivery_data: Dict containing:
                - distance_km: float
                - package_weight_kg: float
                - vehicle_type: str
                - traffic_level: str
                - weather_condition: str
                - road_type: str
                
        Returns:
            Dict with prediction results
        """
        # Preprocess
        X = self.preprocess(delivery_data)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        result = {
            'delayed': bool(prediction),
            'delay_probability': round(float(probabilities[1]), 4),
            'on_time_probability': round(float(probabilities[0]), 4),
            'risk_level': self._get_risk_level(probabilities[1]),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log prediction
        self._log_prediction(delivery_data, result)
        
        return result
    
    def _get_risk_level(self, delay_prob: float) -> str:
        """Classify delay risk level."""
        if delay_prob < 0.3:
            return 'Low'
        elif delay_prob < 0.6:
            return 'Medium'
        elif delay_prob < 0.8:
            return 'High'
        else:
            return 'Critical'
    
    def _log_prediction(self, input_data: Dict, result: Dict):
        """Log prediction to CSV file."""
        log_entry = {**input_data, **result}
        df = pd.DataFrame([log_entry])
        
        # Append to log file
        if os.path.exists(self.log_path):
            df.to_csv(self.log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.log_path, index=False)
    
    def predict_batch(self, deliveries: pd.DataFrame) -> pd.DataFrame:
        """
        Predict delays for multiple deliveries.
        
        Args:
            deliveries: DataFrame with delivery data
            
        Returns:
            DataFrame with predictions added
        """
        results = deliveries.copy()
        predictions = []
        
        for _, row in deliveries.iterrows():
            data = row[self.feature_cols].to_dict()
            pred = self.predict_delay(data)
            predictions.append(pred)
        
        results['delayed_prediction'] = [p['delayed'] for p in predictions]
        results['delay_probability'] = [p['delay_probability'] for p in predictions]
        results['risk_level'] = [p['risk_level'] for p in predictions]
        
        return results
    
    def predict_with_route(self, deliveries: pd.DataFrame, 
                           depot_coords: Optional[tuple] = None) -> Dict:
        """
        Predict delays and optimize route for a batch of deliveries.
        
        Args:
            deliveries: DataFrame with delivery data
            depot_coords: Optional (lat, lon) of starting depot
            
        Returns:
            Dict with predictions and optimized route
        """
        # Predict delays
        predicted = self.predict_batch(deliveries)
        
        # Sort by risk level to prioritize high-risk deliveries
        risk_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        predicted['risk_order'] = predicted['risk_level'].map(risk_order)
        predicted = predicted.sort_values('risk_order')
        
        # Optimize route
        route_data = predicted[['delivery_id', 'dest_lat', 'dest_lng', 'traffic_level']].copy()
        route_result = optimize_deliveries(route_data, depot_coords)
        
        return {
            'predictions': predicted.to_dict('records'),
            'route': route_result,
            'summary': {
                'total_deliveries': len(deliveries),
                'predicted_delays': int(predicted['delayed_prediction'].sum()),
                'high_risk_count': int((predicted['risk_level'].isin(['High', 'Critical'])).sum()),
                'avg_delay_probability': round(predicted['delay_probability'].mean(), 4)
            }
        }


def create_sample_delivery() -> Dict:
    """Create a sample delivery for testing."""
    return {
        'distance_km': np.random.uniform(1, 20),
        'package_weight_kg': np.random.uniform(0.5, 30),
        'vehicle_type': np.random.choice(['Bike', 'Van', 'Truck', 'Electric']),
        'traffic_level': np.random.choice(['Low', 'Medium', 'High', 'Very High']),
        'weather_condition': np.random.choice(['Clear', 'Cloudy', 'Rainy', 'Foggy']),
        'road_type': np.random.choice(['City', 'Highway', 'Rural'])
    }


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("DELIVERY DELAY PREDICTION PIPELINE")
    print("="*60)
    
    # Initialize predictor
    predictor = DeliveryPredictor()
    
    # Test single prediction
    print("\n--- Single Delivery Prediction ---")
    sample = create_sample_delivery()
    print(f"Input: {sample}")
    
    result = predictor.predict_delay(sample)
    print(f"\nPrediction:")
    print(f"  Delayed: {result['delayed']}")
    print(f"  Delay Probability: {result['delay_probability']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")
    
    # Test batch prediction with route
    print("\n--- Batch Prediction with Route Optimization ---")
    df = pd.read_csv('prepared_logistics_dataset.csv')
    sample_batch = df.head(5)
    
    batch_result = predictor.predict_with_route(sample_batch)
    
    print(f"\nSummary:")
    print(f"  Total deliveries: {batch_result['summary']['total_deliveries']}")
    print(f"  Predicted delays: {batch_result['summary']['predicted_delays']}")
    print(f"  High risk count: {batch_result['summary']['high_risk_count']}")
    print(f"  Avg delay probability: {batch_result['summary']['avg_delay_probability']:.2%}")
    
    print(f"\nOptimized Route:")
    print(f"  Distance: {batch_result['route']['total_distance_km']} km")
    print(f"  Estimated time: {batch_result['route']['estimated_time_hours']} hours")
    
    print("\n" + "="*60)
    print("Predictions logged to: logs/predictions.csv")
