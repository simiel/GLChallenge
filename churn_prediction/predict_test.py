import os
import pandas as pd
from main import predict_pipeline
import datetime

def run_predictions(
    test_data_path: str,
    model_dir: str = 'models',
    output_dir: str = 'predictions'
) -> str:
    """
    Run predictions on test data and save results to CSV
    Returns the path to the saved predictions file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run predictions
    print(f"Running predictions on {test_data_path}...")
    results, artifacts = predict_pipeline(test_data_path, model_dir)
    
    # Add original data to results
    original_data = pd.read_csv(test_data_path)
    final_results = pd.concat([original_data, results], axis=1)
    
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'churn_predictions_{timestamp}.csv')
    
    # Save predictions
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
    
    parser = argparse.ArgumentParser(description='Run Churn Predictions on Test Data')
    parser.add_argument('--test_data', required=True,
                      help='Path to the test data CSV file')
    parser.add_argument('--model_dir', default='models',
                      help='Directory containing the trained model')
    parser.add_argument('--output_dir', default='predictions',
                      help='Directory to save prediction results')
    
    args = parser.parse_args()
    
    output_file = run_predictions(
        args.test_data,
        args.model_dir,
        args.output_dir
    ) 