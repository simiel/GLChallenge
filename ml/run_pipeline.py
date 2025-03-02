import argparse
import os
import sys
from preprocessing.preprocessor import run_preprocessing
from training.clusterer import run_clustering
from config import *

def main(input_file):
    """Run the complete customer segmentation pipeline."""
    try:
        print("\n=== Starting Customer Segmentation Pipeline ===\n")
        
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Step 1: Preprocessing
        print("Step 1: Preprocessing")
        print("-" * 50)
        preprocess_stats = run_preprocessing(input_file)
        print(f"\nPreprocessing complete:")
        print(f"- Feature matrix shape: {preprocess_stats['X_shape']}")
        print(f"- Number of customers: {preprocess_stats['n_customers']}")
        
        # Step 2: Clustering
        print("\nStep 2: Clustering")
        print("-" * 50)
        cluster_stats = run_clustering()
        
        print(f"\nClustering complete:")
        print(f"- Number of clusters: {cluster_stats['n_clusters']}")
        print("\nSilhouette scores:")
        for n_clusters, score in sorted(cluster_stats['silhouette_scores'].items()):
            print(f"- {n_clusters} clusters: {score:.3f}")
        print("\nCluster sizes:")
        for cluster, size in sorted(cluster_stats['cluster_sizes'].items()):
            print(f"- Cluster {cluster}: {size} customers")
        
        print("\n=== Pipeline Complete ===")
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