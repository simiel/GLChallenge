import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class CustomerSegmenter:
    def __init__(self, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
        """Initialize the customer segmenter with the specified number of clusters."""
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.silhouette_scores = None
        
    def find_optimal_clusters(self, X, max_clusters=10):
        """Find the optimal number of clusters using silhouette score."""
        print("Finding optimal number of clusters...")
        silhouette_scores = []
        n_clusters_range = range(2, max_clusters + 1)
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"Silhouette score for {n_clusters} clusters: {silhouette_avg:.3f}")
        
        self.silhouette_scores = dict(zip(n_clusters_range, silhouette_scores))
        optimal_n_clusters = max(self.silhouette_scores.items(), key=lambda x: x[1])[0]
        print(f"\nOptimal number of clusters: {optimal_n_clusters}")
        return optimal_n_clusters
    
    def fit(self, X, find_optimal=True):
        """Fit the clustering model to the data."""
        print("Fitting clustering model...")
        if find_optimal:
            self.n_clusters = self.find_optimal_clusters(X)
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans.fit(X)
        return self
    
    def predict(self, X):
        """Predict cluster labels for X."""
        if self.kmeans is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.kmeans.predict(X)
    
    def fit_predict(self, X, find_optimal=True):
        """Fit the model and predict cluster labels."""
        return self.fit(X, find_optimal).predict(X)
    
    def get_cluster_centers(self):
        """Get the cluster centers."""
        if self.kmeans is None:
            raise ValueError("Model must be fitted before getting cluster centers")
        return self.kmeans.cluster_centers_
    
    def get_silhouette_scores(self):
        """Get the silhouette scores for different numbers of clusters."""
        if self.silhouette_scores is None:
            raise ValueError("Must run find_optimal_clusters first to get silhouette scores")
        return self.silhouette_scores
    
    def save_model(self, model_path):
        """Save the trained model."""
        if self.kmeans is None:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.kmeans, model_path)
        print(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path):
        """Load a trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create a new instance
        instance = cls()
        
        # Load the model
        instance.kmeans = joblib.load(model_path)
        instance.n_clusters = instance.kmeans.n_clusters
        
        return instance

def run_clustering(X=None, customer_ids=None):
    """Run the clustering pipeline."""
    # Load data if not provided
    if X is None or customer_ids is None:
        print("Loading preprocessed data...")
        X = np.load(os.path.join(PROCESSED_DATA_DIR, PROCESSED_FEATURES_FILE), allow_pickle=True)
        customer_ids = np.load(os.path.join(PROCESSED_DATA_DIR, CUSTOMER_IDS_FILE), allow_pickle=True)
    
    # Initialize and train the model
    segmenter = CustomerSegmenter()
    cluster_labels = segmenter.fit_predict(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'CustomerID': customer_ids,
        'Cluster': cluster_labels
    })
    
    # Save model and results
    os.makedirs(MODEL_DIR, exist_ok=True)
    segmenter.save_model(os.path.join(MODEL_DIR, MODEL_FILE))
    
    results_path = os.path.join(RESULTS_DIR, CLUSTER_ASSIGNMENTS_FILE)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(results_path, index=False)
    
    print(f"\nClustering complete!")
    print(f"Number of clusters: {segmenter.n_clusters}")
    print(f"Results saved to {results_path}")
    
    return {
        'n_clusters': segmenter.n_clusters,
        'silhouette_scores': segmenter.get_silhouette_scores(),
        'cluster_sizes': results['Cluster'].value_counts().to_dict()
    }

if __name__ == "__main__":
    run_clustering() 