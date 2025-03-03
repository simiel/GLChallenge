import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import joblib
import json
from datetime import datetime

from data_preprocessing import load_data, transform_new_data

class PredictionPipeline:
    def __init__(self, model_dir: str = 'models'):
        """Initialize the prediction pipeline"""
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist. Please train the model first.")
    
    def _load_artifacts(self, name: str) -> Dict:
        """Helper method to load artifacts"""
        path = os.path.join(self.model_dir, f'{name}.joblib')
        if not os.path.exists(path):
            raise ValueError(f"Required artifact {name} not found in {self.model_dir}")
        return joblib.load(path)
    
    def _load_json(self, name: str) -> Dict:
        """Helper method to load JSON files"""
        path = os.path.join(self.model_dir, f'{name}.json')
        if not os.path.exists(path):
            raise ValueError(f"Required file {name}.json not found in {self.model_dir}")
        with open(path, 'r') as f:
            return json.load(f)
    
    def _validate_input_data(self, df: pd.DataFrame, feature_config: Dict) -> List[str]:
        """Validate input data against training feature configuration"""
        missing_cols = []
        for col_type in ['numerical', 'categorical']:
            for col in feature_config[col_type]:
                if col not in df.columns:
                    missing_cols.append(col)
        return missing_cols
    
    def predict(self, data_path: str, output_dir: str = 'predictions') -> Tuple[str, pd.DataFrame]:
        """Make predictions using saved model and artifacts"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n=== Starting Prediction Pipeline ===")
        
        # Step 1: Load all necessary artifacts
        print("\nStep 1: Loading model and artifacts...")
        try:
            # Load configurations and artifacts
            feature_config = self._load_json('feature_config')
            pipeline_metadata = self._load_json('pipeline_metadata')
            model_metrics = self._load_json('model_metrics')
            preprocessing_artifacts = self._load_artifacts('preprocessing')
            model = joblib.load(os.path.join(self.model_dir, 'model.joblib'))
            
            print("Model Information:")
            print(f"Training Date: {pipeline_metadata['training_date']}")
            print(f"Model Type: {pipeline_metadata['model_type']}")
            print("Training Metrics:")
            for metric, value in pipeline_metadata['metrics'].items():
                print(f"- {metric.upper()}: {value:.4f}")
        except Exception as e:
            raise Exception(f"Error loading model artifacts: {str(e)}")
        
        # Step 2: Load and validate input data
        print("\nStep 2: Loading and validating input data...")
        df = load_data(data_path)
        print(f"Input data shape: {df.shape}")
        
        # Validate required columns
        missing_cols = self._validate_input_data(df, feature_config)
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {', '.join(missing_cols)}")
        
        # Step 3: Preprocess input data
        print("\nStep 3: Preprocessing input data...")
        df_processed = transform_new_data(df, preprocessing_artifacts)
        print(f"Processed data shape: {df_processed.shape}")
        
        # Step 4: Generate predictions
        print("\nStep 4: Generating predictions...")
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'churn_prediction': predictions,
            'churn_probability': probabilities
        }, index=df.index)
        
        # Add metadata
        results['model_version'] = pipeline_metadata['training_date']
        results['prediction_date'] = datetime.now().isoformat()
        
        # Combine with original data
        final_results = pd.concat([df, results], axis=1)
        
        # Save predictions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'churn_predictions_{timestamp}.csv')
        final_results.to_csv(output_file, index=False)
        
        # Generate prediction summary
        print("\n=== Prediction Summary ===")
        print(f"\nPredictions saved to: {output_file}")
        print("\nChurn Risk Distribution:")
        print("-" * 50)
        churn_dist = results['churn_prediction'].value_counts(normalize=True).round(3)
        print("Non-churners (0):", churn_dist.get(0, 0))
        print("Churners (1):", churn_dist.get(1, 0))
        
        print("\nChurn Probability Statistics:")
        print("-" * 50)
        prob_stats = results['churn_probability'].describe().round(3)
        print(f"Mean: {prob_stats['mean']:.3f}")
        print(f"Std Dev: {prob_stats['std']:.3f}")
        print(f"Min: {prob_stats['min']:.3f}")
        print(f"25th percentile: {prob_stats['25%']:.3f}")
        print(f"Median: {prob_stats['50%']:.3f}")
        print(f"75th percentile: {prob_stats['75%']:.3f}")
        print(f"Max: {prob_stats['max']:.3f}")
        
        return output_file, final_results

    def predict_dataframe(self, df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
        """Make predictions on a DataFrame directly"""
        print("\n=== Starting Prediction Pipeline ===")
        
        # Step 1: Load all necessary artifacts
        print("\nStep 1: Loading model and artifacts...")
        try:
            # Load configurations and artifacts
            feature_config = self._load_json('feature_config')
            pipeline_metadata = self._load_json('pipeline_metadata')
            model_metrics = self._load_json('model_metrics')
            preprocessing_artifacts = self._load_artifacts('preprocessing')
            model = joblib.load(os.path.join(self.model_dir, 'model.joblib'))
        except Exception as e:
            raise Exception(f"Error loading model artifacts: {str(e)}")
        
        # Step 2: Validate input data
        print("\nStep 2: Validating input data...")
        print(f"Input data shape: {df.shape}")
        
        # Validate required columns
        missing_cols = self._validate_input_data(df, feature_config)
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {', '.join(missing_cols)}")
        
        # Step 3: Preprocess input data
        print("\nStep 3: Preprocessing input data...")
        df_processed = transform_new_data(df, preprocessing_artifacts)
        print(f"Processed data shape: {df_processed.shape}")
        
        # Step 4: Generate predictions
        print("\nStep 4: Generating predictions...")
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'churn_prediction': predictions,
            'churn_probability': probabilities
        }, index=df.index)
        
        # Add metadata
        results['model_version'] = pipeline_metadata['training_date']
        results['prediction_date'] = datetime.now().isoformat()
        
        # Combine with original data
        final_results = pd.concat([df, results], axis=1)
        
        return None, final_results  # Return None for file path since we're not saving

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Churn Predictions')
    parser.add_argument('--data_path', required=True,
                      help='Path to the input data CSV file')
    parser.add_argument('--model_dir', default='models',
                      help='Directory containing model artifacts')
    parser.add_argument('--output_dir', default='predictions',
                      help='Directory to save prediction results')
    
    args = parser.parse_args()
    
    pipeline = PredictionPipeline(model_dir=args.model_dir)
    output_file, _ = pipeline.predict(args.data_path, args.output_dir) 