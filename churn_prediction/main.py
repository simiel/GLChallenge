import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import json

from data_preprocessing import (
    load_data,
    prepare_features,
    transform_new_data
)
from unsupervised_churn import generate_ensemble_churn_labels
from supervised_model import (
    train_model,
    save_model,
    load_model,
    predict_churn,
    evaluate_predictions
)

def create_behavior_weights(feature_columns: Dict[str, list]) -> Dict[str, float]:
    """
    Create behavior weights for clustering based on domain knowledge
    """
    weights = {}
    for i, col in enumerate(feature_columns['numerical'] + feature_columns['categorical']):
        # Example: Give higher weights to transaction-related features
        if 'transaction' in col.lower():
            weights[i] = 1.5
        elif 'balance' in col.lower():
            weights[i] = 1.2
        else:
            weights[i] = 1.0
    return weights

def train_pipeline(
    data_path: str,
    model_dir: str = 'models',
    random_state: int = 42
) -> Dict:
    """
    Run the complete training pipeline
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(data_path)
    
    # Define feature columns (customize based on your data)
    feature_columns = {
        'numerical': [col for col in df.columns if df[col].dtype in ['int64', 'float64']],
        'categorical': [col for col in df.columns if df[col].dtype == 'object']
    }
    
    # Prepare features
    df_processed, preprocessing_artifacts = prepare_features(
        df,
        feature_columns['numerical'],
        feature_columns['categorical']
    )
    
    # Generate churn labels using unsupervised learning
    print("Generating churn labels...")
    behavior_weights = create_behavior_weights(feature_columns)
    churn_labels, clustering_artifacts = generate_ensemble_churn_labels(
        df_processed,
        behavior_weights=behavior_weights,
        random_state=random_state
    )
    
    # Train supervised model
    print("Training supervised model...")
    model, model_artifacts = train_model(
        df_processed,
        churn_labels,
        model_type='rf',
        random_state=random_state
    )
    
    # Save artifacts
    artifacts = {
        'preprocessing': preprocessing_artifacts,
        'clustering': clustering_artifacts,
        'model': model_artifacts
    }
    
    # Save model and artifacts
    save_model(
        model,
        artifacts,
        os.path.join(model_dir, 'model.joblib'),
        os.path.join(model_dir, 'artifacts.joblib')
    )
    
    # Save feature columns configuration
    with open(os.path.join(model_dir, 'feature_columns.json'), 'w') as f:
        json.dump(feature_columns, f)
    
    print("Training complete!")
    return artifacts

def predict_pipeline(
    data_path: str,
    model_dir: str = 'models'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the complete prediction pipeline for new data
    """
    # Load model and artifacts
    model, artifacts = load_model(
        os.path.join(model_dir, 'model.joblib'),
        os.path.join(model_dir, 'artifacts.joblib')
    )
    
    # Load feature columns configuration
    with open(os.path.join(model_dir, 'feature_columns.json'), 'r') as f:
        feature_columns = json.load(f)
    
    # Load and preprocess new data
    df = load_data(data_path)
    
    # Transform features using saved preprocessors
    df_processed = transform_new_data(df, artifacts['preprocessing'])
    
    # Make predictions
    predictions, probabilities = predict_churn(model, df_processed)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'churn_prediction': predictions,
        'churn_probability': probabilities
    }, index=df.index)
    
    return results, artifacts

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Churn Prediction Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Pipeline mode: train or predict')
    parser.add_argument('--data_path', required=True,
                      help='Path to the input data CSV file')
    parser.add_argument('--model_dir', default='models',
                      help='Directory for model artifacts')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        artifacts = train_pipeline(
            args.data_path,
            args.model_dir,
            args.random_state
        )
        print("\nTraining metrics:")
        print(json.dumps(artifacts['model']['metrics'], indent=2))
        
    else:  # predict
        results, artifacts = predict_pipeline(
            args.data_path,
            args.model_dir
        )
        print("\nPrediction results:")
        print(results.describe())
        print("\nChurn distribution:")
        print(results['churn_prediction'].value_counts(normalize=True)) 