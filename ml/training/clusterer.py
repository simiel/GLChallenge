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

def load_preprocessed_data(data_dir='../data/processed'):
    """Load preprocessed data with feature names."""
    X = np.load(f'{data_dir}/X_processed.npy')
    customer_ids = np.load(f'{data_dir}/customer_ids.npy')
    feature_names = np.load(f'{data_dir}/feature_names.npy')
    return X, customer_ids, feature_names

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
    plt.figure(figsize=(15, 5))
    
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
    return optimal_k, inertias, silhouette_scores

def analyze_clusters(X, cluster_labels, customer_ids, feature_names):
    """Detailed analysis of cluster characteristics."""
    # Create output directory if it doesn't exist
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame with features and cluster labels
    df = pd.DataFrame(X, columns=feature_names)
    df['Cluster'] = cluster_labels
    df['CustomerID'] = customer_ids
    
    # Calculate cluster profiles (exclude categorical columns for mean calculation)
    numerical_features = [col for col in feature_names if col not in CATEGORICAL_COLUMNS]
    cluster_profiles = df.groupby('Cluster')[numerical_features].mean()
    
    # Calculate relative characteristics (z-scores) for numerical features
    cluster_zscores = pd.DataFrame()
    for feature in numerical_features:
        mean = df[feature].mean()
        std = df[feature].std()
        cluster_zscores[feature] = (cluster_profiles[feature] - mean) / std
    
    # Add categorical feature distributions
    for cat_feature in CATEGORICAL_COLUMNS:
        if cat_feature in feature_names:
            # Calculate distribution of categories in each cluster
            cat_dist = df.groupby('Cluster')[cat_feature].value_counts(normalize=True).unstack()
            cluster_profiles = pd.concat([cluster_profiles, cat_dist], axis=1)
    
    # Identify distinguishing features for each cluster
    distinguishing_features = {}
    for cluster in cluster_profiles.index:
        # Get top 5 highest and lowest z-scores for numerical features
        cluster_scores = cluster_zscores.loc[cluster].sort_values()
        distinguishing_features[cluster] = {
            'highest': cluster_scores[-5:].index.tolist(),
            'lowest': cluster_scores[:5].index.tolist()
        }
        
        # Add categorical feature insights if they exist
        for cat_feature in CATEGORICAL_COLUMNS:
            if cat_feature in feature_names:
                cat_dist = df[df['Cluster'] == cluster][cat_feature].value_counts(normalize=True)
                distinguishing_features[cluster][f'{cat_feature}_dist'] = cat_dist.to_dict()
    
    # Visualizations
    
    # 1. PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('Customer Segments (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.savefig(os.path.join(output_dir, 'cluster_pca_visualization.png'))
    plt.close()
    
    # 2. Feature distributions by cluster (numerical features)
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numerical_features[:5]):  # Plot first 5 numerical features
        plt.subplot(2, 3, i+1)
        sns.boxplot(x='Cluster', y=feature, data=df)
        plt.xticks(rotation=45)
        plt.title(f'{feature} by Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_feature_distributions.png'))
    plt.close()
    
    # 3. Cluster sizes
    plt.figure(figsize=(10, 6))
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    cluster_sizes.plot(kind='bar')
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    plt.savefig(os.path.join(output_dir, 'cluster_sizes.png'))
    plt.close()
    
    # 4. Heatmap of cluster profiles (numerical features)
    plt.figure(figsize=(15, 8))
    sns.heatmap(cluster_zscores, cmap='RdYlBu', center=0, annot=True, fmt='.2f')
    plt.title('Cluster Characteristics (Z-scores)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_heatmap.png'))
    plt.close()
    
    # Save detailed analysis
    with open(os.path.join(output_dir, 'cluster_analysis.txt'), 'w') as f:
        f.write("Cluster Analysis Report\n")
        f.write("=====================\n\n")
        
        # Cluster sizes
        f.write("Cluster Sizes:\n")
        f.write("--------------\n")
        for cluster, size in cluster_sizes.items():
            f.write(f"Cluster {cluster}: {size} customers ({size/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        # Cluster profiles
        f.write("Cluster Profiles:\n")
        f.write("----------------\n")
        f.write(cluster_profiles.to_string())
        f.write("\n\n")
        
        # Distinguishing features
        f.write("Distinguishing Features by Cluster:\n")
        f.write("--------------------------------\n")
        for cluster, features in distinguishing_features.items():
            f.write(f"\nCluster {cluster}:\n")
            f.write("  High values in:\n")
            for feature in features['highest']:
                zscore = cluster_zscores.loc[cluster, feature]
                f.write(f"    - {feature}: {zscore:.2f} std. dev. from mean\n")
            f.write("  Low values in:\n")
            for feature in features['lowest']:
                zscore = cluster_zscores.loc[cluster, feature]
                f.write(f"    - {feature}: {zscore:.2f} std. dev. from mean\n")
            
            # Add categorical distributions if they exist
            for cat_feature in CATEGORICAL_COLUMNS:
                if f'{cat_feature}_dist' in features:
                    f.write(f"  {cat_feature} distribution:\n")
                    for category, prop in features[f'{cat_feature}_dist'].items():
                        f.write(f"    - {category}: {prop:.1%}\n")
    
    return {
        'cluster_profiles': cluster_profiles,
        'cluster_zscores': cluster_zscores,
        'distinguishing_features': distinguishing_features,
        'cluster_sizes': cluster_sizes
    }

def identify_churn_clusters(cluster_profiles, cluster_zscores):
    """Identify potential churn clusters based on characteristics."""
    # Calculate risk scores for each cluster based on churn-indicating features
    risk_scores = pd.Series(index=cluster_profiles.index, dtype=float)
    
    # Features that might indicate churn risk (customize based on domain knowledge)
    churn_indicators = {
        'transaction_frequency': -1,  # Lower is riskier
        'total_transaction_amount': -1,  # Lower is riskier
        'days_since_last_transaction': 1,  # Higher is riskier
        'transaction_count': -1,  # Lower is riskier
        'avg_transaction_amount': -1  # Lower is riskier
    }
    
    # Calculate risk score for each cluster
    for cluster in cluster_profiles.index:
        score = 0
        for feature, direction in churn_indicators.items():
            if feature in cluster_zscores.columns:
                score += cluster_zscores.loc[cluster, feature] * direction
        risk_scores[cluster] = score
    
    # Sort clusters by risk score
    risk_scores = risk_scores.sort_values(ascending=False)
    
    return risk_scores

def save_analysis_results(analysis_results, risk_scores, output_dir='../outputs'):
    """Save detailed analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cluster profiles
    analysis_results['cluster_profiles'].to_csv(f'{output_dir}/cluster_profiles.csv')
    
    # Save detailed analysis report
    with open(f'{output_dir}/cluster_analysis_report.txt', 'w') as f:
        f.write("Cluster Analysis Report\n")
        f.write("=====================\n\n")
        
        # Cluster sizes
        f.write("Cluster Sizes:\n")
        f.write("--------------\n")
        for cluster, size in analysis_results['cluster_sizes'].items():
            f.write(f"Cluster {cluster}: {size} customers ({size/sum(analysis_results['cluster_sizes'])*100:.1f}%)\n")
        f.write("\n")
        
        # Distinguishing features
        f.write("Distinguishing Features by Cluster:\n")
        f.write("--------------------------------\n")
        for cluster, features in analysis_results['distinguishing_features'].items():
            f.write(f"\nCluster {cluster}:\n")
            f.write("  High values in:\n")
            for feature in features['highest']:
                zscore = analysis_results['cluster_zscores'].loc[cluster, feature]
                f.write(f"    - {feature}: {zscore:.2f} std. dev. from mean\n")
            f.write("  Low values in:\n")
            for feature in features['lowest']:
                zscore = analysis_results['cluster_zscores'].loc[cluster, feature]
                f.write(f"    - {feature}: {zscore:.2f} std. dev. from mean\n")
        
        # Churn risk analysis
        f.write("\nChurn Risk Analysis:\n")
        f.write("------------------\n")
        for cluster, score in risk_scores.items():
            f.write(f"Cluster {cluster}: Risk Score = {score:.2f}\n")
        
        # Recommendations
        f.write("\nRecommendations:\n")
        f.write("---------------\n")
        high_risk_clusters = risk_scores.head(2).index.tolist()
        f.write(f"Based on the analysis, clusters {high_risk_clusters} show the highest risk of churn.\n")
        f.write("These clusters should be labeled as the churn class for supervised learning.\n")

if __name__ == "__main__":
    # Load data
    print("Loading preprocessed data...")
    X, customer_ids, feature_names = load_preprocessed_data()
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    optimal_k, inertias, silhouette_scores = find_optimal_clusters(X)
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Train clustering model
    print("Training clustering model...")
    model = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = model.fit_predict(X)
    
    # Analyze clusters
    print("Analyzing clusters...")
    analysis_results = analyze_clusters(X, cluster_labels, customer_ids, feature_names)
    
    # Identify potential churn clusters
    print("Identifying potential churn clusters...")
    risk_scores = identify_churn_clusters(
        analysis_results['cluster_profiles'],
        analysis_results['cluster_zscores']
    )
    
    # Save results
    print("Saving analysis results...")
    save_analysis_results(analysis_results, risk_scores)
    
    # Save model
    joblib.dump(model, '../models/kmeans_model.joblib')
    
    print("Cluster analysis complete!") 