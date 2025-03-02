import os
import pandas as pd
import numpy as np
from typing import Dict
import joblib
import json
from datetime import datetime

from data_preprocessing import load_data, prepare_features
from unsupervised_churn import generate_ensemble_churn_labels
from supervised_model import train_model

class TrainingPipeline:
    def __init__(self, model_dir: str = 'models'):
        """Initialize the training pipeline"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def _save_artifacts(self, artifacts: Dict, name: str):
        """Helper method to save artifacts"""
        joblib.dump(artifacts, os.path.join(self.model_dir, f'{name}.joblib'))
    
    def train(self, data_path: str, random_state: int = 42) -> Dict:
        """Train the complete pipeline and save all necessary artifacts"""
        print("\n=== Starting Training Pipeline ===")
        print("\nStep 1: Loading and preprocessing data...")
        df = load_data(data_path)
        
        # Define and save feature columns
        feature_columns = {
            'numerical': [col for col in df.columns if df[col].dtype in ['int64', 'float64']],
            'categorical': [col for col in df.columns if df[col].dtype == 'object']
        }
        
        print(f"Found {len(feature_columns['numerical'])} numerical and {len(feature_columns['categorical'])} categorical features")
        
        # Save feature configuration
        with open(os.path.join(self.model_dir, 'feature_config.json'), 'w') as f:
            json.dump(feature_columns, f)
        
        # Step 1: Prepare features and save preprocessing artifacts
        print("\nStep 2: Preparing and transforming features...")
        df_processed, preprocessing_artifacts = prepare_features(
            df,
            feature_columns['numerical'],
            feature_columns['categorical']
        )
        self._save_artifacts(preprocessing_artifacts, 'preprocessing')
        print(f"Processed data shape: {df_processed.shape}")
        
        # Step 2: Generate churn labels and save clustering artifacts
        print("\nStep 3: Generating churn labels using unsupervised learning...")
        behavior_weights = {i: 1.0 for i in range(df_processed.shape[1])}
        churn_labels, clustering_artifacts = generate_ensemble_churn_labels(
            df_processed,
            behavior_weights=behavior_weights,
            random_state=random_state
        )
        self._save_artifacts(clustering_artifacts, 'clustering')
        print(f"Generated labels distribution: {np.bincount(churn_labels)}")
        
        # Step 3: Train supervised model and save model artifacts
        print("\nStep 4: Training supervised model...")
        model, model_artifacts = train_model(
            df_processed,
            churn_labels,
            model_type='rf',
            random_state=random_state
        )
        
        # Save model and metrics separately
        joblib.dump(model, os.path.join(self.model_dir, 'model.joblib'))
        with open(os.path.join(self.model_dir, 'model_metrics.json'), 'w') as f:
            json.dump(model_artifacts, f)
        
        # Save pipeline metadata
        pipeline_metadata = {
            'training_date': datetime.now().isoformat(),
            'data_path': data_path,
            'random_state': random_state,
            'feature_columns': feature_columns,
            'model_type': 'rf',
            'metrics': model_artifacts['metrics'],
            'feature_importance': model_artifacts['feature_importance']
        }
        with open(os.path.join(self.model_dir, 'pipeline_metadata.json'), 'w') as f:
            json.dump(pipeline_metadata, f)
        
        print("\n=== Training Complete ===")
        print(f"All artifacts saved in: {self.model_dir}")
        print("\nModel Performance Metrics:")
        print("-" * 50)
        for metric, value in model_artifacts['metrics'].items():
            print(f"{metric.upper()}: {value:.4f}")
        
        print("\nTop 10 Most Important Features:")
        print("-" * 50)
        features = model_artifacts['feature_importance']['features']
        importance = model_artifacts['feature_importance']['importance']
        feature_imp = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
        for feature, imp in feature_imp[:10]:
            print(f"{feature}: {imp:.4f}")
        
        return pipeline_metadata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Churn Prediction Model')
    parser.add_argument('--data_path', required=True,
                      help='Path to the training data CSV file')
    parser.add_argument('--model_dir', default='models',
                      help='Directory to save model artifacts')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(model_dir=args.model_dir)
    pipeline.train(args.data_path, args.random_state) 