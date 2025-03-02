import os
import pandas as pd
import numpy as np
import joblib
from data_preprocessing import transform_new_data

def predict_on_new_data(
    test_data_path: str,
    model_path: str = 'models/model.joblib',
    artifacts_path: str = 'models/artifacts.joblib',
    output_dir: str = 'predictions'
) -> str:
    """
    Load saved model and make predictions on new data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model and artifacts
    print("Loading saved model and artifacts...")
    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    
    # Load and preprocess test data
    print(f"Processing test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    
    # Transform features using saved preprocessing artifacts
    test_processed = transform_new_data(test_data, artifacts['preprocessing'])
    
    # Make predictions
    print("Generating predictions...")
    predictions = model.predict(test_processed)
    probabilities = model.predict_proba(test_processed)[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'churn_prediction': predictions,
        'churn_probability': probabilities
    }, index=test_data.index)
    
    # Combine with original data
    final_results = pd.concat([test_data, results], axis=1)
    
    # Save predictions
    output_file = os.path.join(output_dir, 'churn_predictions.csv')
    final_results.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print("-" * 50)
    print("Churn Risk Distribution:")
    print(results['churn_prediction'].value_counts(normalize=True).round(3))
    print("\nChurn Probability Statistics:")
    print(results['churn_probability'].describe().round(3))
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions using saved churn model')
    parser.add_argument('--test_data', required=True,
                      help='Path to the test data CSV file')
    parser.add_argument('--model_path', default='models/model.joblib',
                      help='Path to the saved model file')
    parser.add_argument('--artifacts_path', default='models/artifacts.joblib',
                      help='Path to the saved artifacts file')
    parser.add_argument('--output_dir', default='predictions',
                      help='Directory to save prediction results')
    
    args = parser.parse_args()
    
    output_file = predict_on_new_data(
        args.test_data,
        args.model_path,
        args.artifacts_path,
        args.output_dir
    ) 