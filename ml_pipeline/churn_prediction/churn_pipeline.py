import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import joblib
import json
from datetime import datetime

from data_preprocessing import (
    load_data,
    prepare_features,
    transform_new_data
)
from unsupervised_churn import generate_ensemble_churn_labels
from supervised_model import train_model

class ChurnPipeline:
    def __init__(self, model_dir: str = 'models'):
        """Initialize the pipeline with model directory"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def _save_artifacts(self, artifacts: Dict, name: str):
        """Helper method to save artifacts"""
        joblib.dump(artifacts, os.path.join(self.model_dir, f'{name}.joblib'))
    
    def _load_artifacts(self, name: str) -> Dict:
        """Helper method to load artifacts"""
        return joblib.load(os.path.join(self.model_dir, f'{name}.joblib'))
        
    def train(self, data_path: str, random_state: int = 42) -> Dict:
        """Train the complete pipeline and save all necessary artifacts"""
        print("Loading and preprocessing data...")
        df = load_data(data_path)
        
        # Define and save feature columns
        feature_columns = {
            'numerical': [col for col in df.columns if df[col].dtype in ['int64', 'float64']],
            'categorical': [col for col in df.columns if df[col].dtype == 'object']
        }
        
        # Save feature configuration
        with open(os.path.join(self.model_dir, 'feature_config.json'), 'w') as f:
            json.dump(feature_columns, f)
        
        # Step 1: Prepare features and save preprocessing artifacts
        print("Preparing features...")
        df_processed, preprocessing_artifacts = prepare_features(
            df,
            feature_columns['numerical'],
            feature_columns['categorical']
        )
        self._save_artifacts(preprocessing_artifacts, 'preprocessing')
        
        # Step 2: Generate churn labels and save clustering artifacts
        print("Generating churn labels...")
        behavior_weights = {i: 1.0 for i in range(df_processed.shape[1])}
        churn_labels, clustering_artifacts = generate_ensemble_churn_labels(
            df_processed,
            behavior_weights=behavior_weights,
            random_state=random_state
        )
        self._save_artifacts(clustering_artifacts, 'clustering')
        
        # Step 3: Train supervised model and save model artifacts
        print("Training supervised model...")
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
            'metrics': model_artifacts['metrics']
        }
        with open(os.path.join(self.model_dir, 'pipeline_metadata.json'), 'w') as f:
            json.dump(pipeline_metadata, f)
        
        print("\nTraining complete! All artifacts saved in:", self.model_dir)
        print("\nModel Metrics:")
        print(json.dumps(model_artifacts['metrics'], indent=2))
        
        return pipeline_metadata
    
    def predict(self, data_path: str, output_dir: str = 'predictions') -> str:
        """Make predictions using saved model and artifacts"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Load all necessary artifacts
        print("Loading model and artifacts...")
        try:
            # Load feature configuration
            with open(os.path.join(self.model_dir, 'feature_config.json'), 'r') as f:
                feature_columns = json.load(f)
            
            # Load preprocessing artifacts
            preprocessing_artifacts = self._load_artifacts('preprocessing')
            
            # Load model
            model = joblib.load(os.path.join(self.model_dir, 'model.joblib'))
            
            # Load pipeline metadata
            with open(os.path.join(self.model_dir, 'pipeline_metadata.json'), 'r') as f:
                pipeline_metadata = json.load(f)
        except FileNotFoundError as e:
            raise Exception("Required model artifacts not found. Please run training first.") from e
        
        # Step 2: Load and preprocess new data
        print("Processing test data...")
        df = load_data(data_path)
        df_processed = transform_new_data(df, preprocessing_artifacts)
        
        # Step 3: Generate predictions
        print("Generating predictions...")
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'churn_prediction': predictions,
            'churn_probability': probabilities
        }, index=df.index)
        
        # Combine with original data
        final_results = pd.concat([df, results], axis=1)
        
        # Add metadata columns
        final_results['model_version'] = pipeline_metadata['training_date']
        final_results['prediction_date'] = datetime.now().isoformat()
        
        # Save predictions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'churn_predictions_{timestamp}.csv')
        final_results.to_csv(output_file, index=False)
        
        print(f"\nPredictions saved to: {output_file}")
        print("\nPrediction Summary:")
        print("-" * 50)
        print("Churn Risk Distribution:")
        print(results['churn_prediction'].value_counts(normalize=True).round(3))
        print("\nChurn Probability Statistics:")
        print(results['churn_probability'].describe().round(3))
        
        return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Churn Prediction Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Pipeline mode: train or predict')
    parser.add_argument('--data_path', required=True,
                      help='Path to the input data CSV file')
    parser.add_argument('--model_dir', default='models',
                      help='Directory for model artifacts')
    parser.add_argument('--output_dir', default='predictions',
                      help='Directory for prediction outputs')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    pipeline = ChurnPipeline(model_dir=args.model_dir)
    
    if args.mode == 'train':
        pipeline.train(args.data_path, args.random_state)
    else:  # predict
        pipeline.predict(args.data_path, args.output_dir) 