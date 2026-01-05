"""
MLOps Automation: Model Retraining
Weekly model retraining with performance logging and versioning.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json


class ModelRetrainer:
    """
    Automated model retraining with versioning and performance tracking.
    """
    
    def __init__(self, data_path: str = 'prepared_logistics_dataset.csv',
                 models_dir: str = 'models',
                 logs_dir: str = 'mlops'):
        self.data_path = data_path
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        self.performance_log = os.path.join(logs_dir, 'performance_log.csv')
        
        # Ensure directories exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Feature configuration
        self.feature_cols = ['distance_km', 'package_weight_kg', 'vehicle_type',
                            'traffic_level', 'weather_condition', 'road_type']
        self.categorical_cols = ['vehicle_type', 'traffic_level', 'weather_condition', 'road_type']
        self.target_col = 'delayed'
    
    def load_data(self) -> pd.DataFrame:
        """Load training data."""
        return pd.read_csv(self.data_path)
    
    def preprocess(self, df: pd.DataFrame) -> tuple:
        """Preprocess data for training."""
        df_model = df[self.feature_cols + [self.target_col]].copy()
        
        # Encode categorical variables
        label_encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            label_encoders[col] = le
        
        X = df_model[self.feature_cols]
        y = df_model[self.target_col]
        
        return X, y, label_encoders
    
    def train_models(self, X_train, y_train) -> dict:
        """Train multiple models."""
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric='logloss'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        }
        
        trained_models = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        return trained_models
    
    def evaluate_models(self, models: dict, X_test, y_test) -> dict:
        """Evaluate all models and return metrics."""
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
        
        return results
    
    def select_best_model(self, models: dict, results: dict) -> tuple:
        """Select best model based on ROC-AUC."""
        best_name = max(results, key=lambda x: results[x]['roc_auc'])
        return best_name, models[best_name]
    
    def save_model(self, model, label_encoders, version: str):
        """Save model with versioning."""
        # Save versioned model
        version_path = os.path.join(self.models_dir, f'model_v{version}.pkl')
        joblib.dump(model, version_path)
        
        # Update best model
        best_path = os.path.join(self.models_dir, 'best_delay_prediction_model.pkl')
        joblib.dump(model, best_path)
        
        # Save encoders
        encoders_path = os.path.join(self.models_dir, 'label_encoders.pkl')
        joblib.dump(label_encoders, encoders_path)
        
        return version_path
    
    def log_performance(self, version: str, best_model_name: str, results: dict):
        """Log training performance."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'version': version,
            'best_model': best_model_name,
            **{f'{name}_{metric}': value 
               for name, metrics in results.items() 
               for metric, value in metrics.items()}
        }
        
        df = pd.DataFrame([log_entry])
        
        if os.path.exists(self.performance_log):
            df.to_csv(self.performance_log, mode='a', header=False, index=False)
        else:
            df.to_csv(self.performance_log, index=False)
        
        # Also save detailed results as JSON
        json_path = os.path.join(self.logs_dir, f'training_v{version}.json')
        with open(json_path, 'w') as f:
            json.dump({
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'best_model': best_model_name,
                'results': results
            }, f, indent=2)
    
    def retrain(self) -> dict:
        """
        Full retraining pipeline.
        
        Returns:
            Dict with training summary
        """
        print("="*60)
        print("MODEL RETRAINING PIPELINE")
        print("="*60)
        
        # Generate version
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"\nVersion: {version}")
        
        # Load and preprocess data
        print("\n1. Loading data...")
        df = self.load_data()
        print(f"   Loaded {len(df)} records")
        
        print("\n2. Preprocessing...")
        X, y, label_encoders = self.preprocess(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train models
        print("\n3. Training models...")
        trained_models = self.train_models(X_train, y_train)
        
        # Evaluate
        print("\n4. Evaluating models...")
        results = self.evaluate_models(trained_models, X_test, y_test)
        
        for name, metrics in results.items():
            print(f"   {name}: ROC-AUC = {metrics['roc_auc']:.4f}")
        
        # Select best
        best_name, best_model = self.select_best_model(trained_models, results)
        print(f"\n5. Best model: {best_name}")
        
        # Save
        print("\n6. Saving model...")
        model_path = self.save_model(best_model, label_encoders, version)
        print(f"   Saved to: {model_path}")
        
        # Log performance
        print("\n7. Logging performance...")
        self.log_performance(version, best_name, results)
        print(f"   Logged to: {self.performance_log}")
        
        print("\n" + "="*60)
        print("RETRAINING COMPLETE")
        print("="*60)
        
        return {
            'version': version,
            'best_model': best_name,
            'metrics': results[best_name],
            'model_path': model_path
        }


def run_weekly_retrain():
    """Entry point for scheduled retraining."""
    retrainer = ModelRetrainer()
    result = retrainer.retrain()
    return result


if __name__ == "__main__":
    run_weekly_retrain()
