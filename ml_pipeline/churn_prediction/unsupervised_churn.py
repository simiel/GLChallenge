import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import Tuple, Dict
from sklearn.metrics import silhouette_score

def identify_churn_cluster(cluster_centers: np.ndarray, behavior_weights: Dict[str, float]) -> int:
    """
    Identify which cluster represents churners based on behavioral patterns
    """
    # Calculate weighted scores for each cluster
    cluster_scores = []
    for center in cluster_centers:
        score = sum(center[i] * weight for i, weight in behavior_weights.items())
        cluster_scores.append(score)
    
    # Cluster with highest risk score is the churn cluster
    return np.argmax(cluster_scores)

def generate_churn_labels_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 2,
    behavior_weights: Dict[str, float] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Generate churn labels using KMeans clustering
    """
    if behavior_weights is None:
        behavior_weights = {i: 1.0 for i in range(df.shape[1])}
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(df)
    
    # Identify churn cluster
    churn_cluster = identify_churn_cluster(kmeans.cluster_centers_, behavior_weights)
    
    # Convert cluster labels to binary churn labels
    churn_labels = (cluster_labels == churn_cluster).astype(int)
    
    # Calculate clustering quality metrics
    silhouette_avg = silhouette_score(df, cluster_labels)
    
    artifacts = {
        'model': kmeans,
        'churn_cluster': churn_cluster,
        'silhouette_score': silhouette_avg,
        'cluster_centers': kmeans.cluster_centers_
    }
    
    return churn_labels, artifacts

def generate_churn_labels_gmm(
    df: pd.DataFrame,
    n_components: int = 2,
    behavior_weights: Dict[str, float] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Generate churn labels using Gaussian Mixture Model
    """
    if behavior_weights is None:
        behavior_weights = {i: 1.0 for i in range(df.shape[1])}
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    cluster_labels = gmm.fit_predict(df)
    
    # Identify churn cluster
    churn_cluster = identify_churn_cluster(gmm.means_, behavior_weights)
    
    # Convert cluster labels to binary churn labels
    churn_labels = (cluster_labels == churn_cluster).astype(int)
    
    # Calculate clustering quality metrics
    silhouette_avg = silhouette_score(df, cluster_labels)
    
    artifacts = {
        'model': gmm,
        'churn_cluster': churn_cluster,
        'silhouette_score': silhouette_avg,
        'cluster_means': gmm.means_
    }
    
    return churn_labels, artifacts

def generate_ensemble_churn_labels(
    df: pd.DataFrame,
    behavior_weights: Dict[str, float] = None,
    threshold: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Generate churn labels using an ensemble of clustering methods
    """
    # Get predictions from both models
    kmeans_labels, kmeans_artifacts = generate_churn_labels_kmeans(
        df, behavior_weights=behavior_weights, random_state=random_state
    )
    gmm_labels, gmm_artifacts = generate_churn_labels_gmm(
        df, behavior_weights=behavior_weights, random_state=random_state
    )
    
    # Combine predictions
    ensemble_probs = (kmeans_labels + gmm_labels) / 2
    ensemble_labels = (ensemble_probs >= threshold).astype(int)
    
    artifacts = {
        'kmeans': kmeans_artifacts,
        'gmm': gmm_artifacts,
        'threshold': threshold
    }
    
    return ensemble_labels, artifacts 