import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
import os

def load_preprocessed_data(data_dir='../data/processed'):
    """Load preprocessed data."""
    X = np.load(f'{data_dir}/X_processed.npy')
    customer_ids = np.load(f'{data_dir}/customer_ids.npy')
    return X, customer_ids

def find_optimal_clusters(X, max_clusters=10):
    """Find optimal number of clusters using elbow method and silhouette score."""
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.savefig('../outputs/cluster_analysis.png')
    plt.close()
    
    # Return optimal k (highest silhouette score)
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_k

def train_clustering_model(X, n_clusters):
    """Train KMeans clustering model."""
    model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = model.fit_predict(X)
    return model, cluster_labels

def analyze_clusters(X, cluster_labels, customer_ids):
    """Analyze and visualize cluster characteristics."""
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create scatter plot of clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('Customer Segments (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.savefig('../outputs/cluster_visualization.png')
    plt.close()
    
    # Create cluster profile summary
    cluster_summary = pd.DataFrame({
        'CustomerID': customer_ids,
        'Cluster': cluster_labels
    })
    
    # Calculate cluster sizes
    cluster_sizes = cluster_summary['Cluster'].value_counts().sort_index()
    
    # Create cluster size visualization
    plt.figure(figsize=(10, 6))
    cluster_sizes.plot(kind='bar')
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    plt.savefig('../outputs/cluster_sizes.png')
    plt.close()
    
    return cluster_summary

def save_model_and_results(model, cluster_summary, output_dir='../outputs', model_dir='../models'):
    """Save the clustering model and results."""
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f'{model_dir}/customer_segmentation.joblib')
    
    # Save cluster summary
    os.makedirs(output_dir, exist_ok=True)
    cluster_summary.to_csv(f'{output_dir}/cluster_assignments.csv', index=False)
    
    # Save cluster profiles
    with open(f'{output_dir}/cluster_profiles.txt', 'w') as f:
        f.write("Customer Segment Profiles:\n\n")
        for cluster in sorted(cluster_summary['Cluster'].unique()):
            cluster_size = len(cluster_summary[cluster_summary['Cluster'] == cluster])
            f.write(f"Cluster {cluster}:\n")
            f.write(f"Size: {cluster_size} customers\n")
            f.write(f"Percentage: {(cluster_size/len(cluster_summary)*100):.2f}%\n\n")

if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    X, customer_ids = load_preprocessed_data()
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    optimal_k = find_optimal_clusters(X)
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Train clustering model
    print("Training clustering model...")
    model, cluster_labels = train_clustering_model(X, optimal_k)
    
    # Analyze clusters
    print("Analyzing clusters...")
    cluster_summary = analyze_clusters(X, cluster_labels, customer_ids)
    
    # Save results
    print("Saving model and results...")
    save_model_and_results(model, cluster_summary)
    
    print("Customer segmentation complete!") 