import argparse
import os
import sys
import numpy as np
import pandas as pd
from preprocessing.preprocessor import CustomerFeatureProcessor
from training.clusterer import analyze_clusters, identify_churn_clusters, save_analysis_results
from sklearn.cluster import KMeans
from config import *

def run_pipeline(input_file):
    """Run the complete pipeline from preprocessing to clustering analysis."""
    
    # Step 1: Preprocessing
    print("\n=== Step 1: Preprocessing ===")
    processor = CustomerFeatureProcessor()
    customer_features = processor.load_and_create_features(input_file)
    
    # Debug: Print feature information
    print("\nFeature Information:")
    print("-" * 50)
    print(customer_features.info())
    print("\nFeature Names:")
    print(customer_features.columns.tolist())
    
    # Preprocess features
    X, customer_ids = processor.preprocess_features(customer_features)
    
    # Debug: Print array information
    print("\nProcessed Data Information:")
    print("-" * 50)
    print(f"X shape: {X.shape}")
    print(f"Number of customers: {len(customer_ids)}")
    
    # Get feature names after preprocessing (to match the actual features)
    feature_names = [col for col in customer_features.columns 
                    if col not in ['CustomerID'] + 
                    [col for col in customer_features.columns if pd.api.types.is_datetime64_any_dtype(customer_features[col])]]
    
    print("\nFeature names for clustering:")
    print(feature_names)
    print(f"Number of features: {len(feature_names)}")
    
    # Save preprocessed data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_processed.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'customer_ids.npy'), customer_ids)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'feature_names.npy'), feature_names)
    
    # Step 2: Clustering Analysis
    print("\n=== Step 2: Clustering Analysis ===")
    
    # Initialize and train KMeans
    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    cluster_labels = model.fit_predict(X)
    
    # Create DataFrame for analysis
    cluster_df = pd.DataFrame(X, columns=feature_names)
    cluster_df['Cluster'] = cluster_labels
    cluster_df['CustomerID'] = customer_ids
    
    # Debug: Print cluster information
    print("\nCluster Information:")
    print("-" * 50)
    print(cluster_df.groupby('Cluster').size())
    
    # Analyze clusters
    print("\nAnalyzing cluster characteristics...")
    analysis_results = analyze_clusters(X, cluster_labels, customer_ids, feature_names)
    
    # Identify potential churn clusters
    print("\nIdentifying potential churn clusters...")
    risk_scores = identify_churn_clusters(
        analysis_results['cluster_profiles'],
        analysis_results['cluster_zscores']
    )
    
    # Save results
    print("\nSaving analysis results...")
    save_analysis_results(analysis_results, risk_scores)
    
    print("\n=== Pipeline Complete ===")
    print(f"Results saved in {OUTPUT_DIR}")
    return analysis_results, risk_scores

def main(input_file):
    """Run the complete customer segmentation pipeline."""
    try:
        print("\n=== Starting Customer Segmentation Pipeline ===\n")
        
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Run the pipeline
        analysis_results, risk_scores = run_pipeline(input_file)
        
        print("\nOutputs:")
        print(f"- Preprocessed data: {PROCESSED_DATA_DIR}")
        print(f"- Model file: {os.path.join(MODEL_DIR, MODEL_FILE)}")
        print(f"- Cluster assignments: {os.path.join(RESULTS_DIR, CLUSTER_ASSIGNMENTS_FILE)}")
        
    except Exception as e:
        print(f"\nError in pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run customer segmentation pipeline')
    parser.add_argument('--input', '-i', required=True,
                      help='Path to input CSV file with transaction data')
    
    args = parser.parse_args()
    main(args.input) 