import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import sys
import joblib

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def load_results():
    """Load clustering results and preprocessed data."""
    # Load cluster assignments
    cluster_assignments = pd.read_csv(os.path.join(RESULTS_DIR, CLUSTER_ASSIGNMENTS_FILE))
    
    # Load preprocessed features and customer IDs
    X = np.load(os.path.join(PROCESSED_DATA_DIR, PROCESSED_FEATURES_FILE), allow_pickle=True)
    customer_ids = np.load(os.path.join(PROCESSED_DATA_DIR, CUSTOMER_IDS_FILE), allow_pickle=True)
    
    # Load preprocessing objects
    scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, SCALER_FILE))
    label_encoders = joblib.load(os.path.join(PROCESSED_DATA_DIR, LABEL_ENCODERS_FILE))
    
    return X, customer_ids, cluster_assignments, scaler, label_encoders

def create_cluster_profile(X, cluster_assignments, scaler):
    """Create detailed profile for each cluster."""
    # Create DataFrame with features
    feature_names = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    features_df = pd.DataFrame(X, columns=feature_names)
    
    # Add cluster assignments
    features_df['Cluster'] = cluster_assignments['Cluster']
    
    # Calculate cluster profiles
    profiles = []
    for cluster in sorted(features_df['Cluster'].unique()):
        cluster_data = features_df[features_df['Cluster'] == cluster]
        
        # Calculate statistics for numerical features
        profile = {
            'Cluster': cluster,
            'Size': len(cluster_data),
            'Percentage': len(cluster_data) / len(features_df) * 100
        }
        
        # Add numerical feature statistics
        for col in NUMERICAL_COLUMNS:
            # Inverse transform the scaled values
            original_values = scaler.inverse_transform(cluster_data[NUMERICAL_COLUMNS])
            col_idx = NUMERICAL_COLUMNS.index(col)
            values = original_values[:, col_idx]
            
            profile.update({
                f'{col}_mean': np.mean(values),
                f'{col}_median': np.median(values),
                f'{col}_std': np.std(values)
            })
        
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

def visualize_clusters(X, cluster_assignments):
    """Create visualizations of the clusters."""
    # Create directory for visualizations
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_assignments['Cluster'], cmap='viridis')
    plt.title('Customer Segments (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_visualization_pca.png'))
    plt.close()
    
    # 2. Cluster sizes
    plt.figure(figsize=(10, 6))
    cluster_sizes = cluster_assignments['Cluster'].value_counts().sort_index()
    cluster_sizes.plot(kind='bar')
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_sizes.png'))
    plt.close()
    
    return pca.explained_variance_ratio_

def save_analysis_results(cluster_profiles, pca_variance_ratio):
    """Save analysis results to files."""
    # Save cluster profiles
    profiles_path = os.path.join(OUTPUT_DIR, 'cluster_profiles.csv')
    cluster_profiles.to_csv(profiles_path, index=False)
    
    # Create detailed text report
    report_path = os.path.join(OUTPUT_DIR, 'cluster_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("Customer Segmentation Analysis Report\n")
        f.write("===================================\n\n")
        
        # PCA results
        f.write("PCA Analysis:\n")
        f.write("--------------\n")
        f.write(f"Variance explained by first two components: {sum(pca_variance_ratio)*100:.2f}%\n")
        f.write(f"- First component: {pca_variance_ratio[0]*100:.2f}%\n")
        f.write(f"- Second component: {pca_variance_ratio[1]*100:.2f}%\n\n")
        
        # Cluster profiles
        f.write("Cluster Profiles:\n")
        f.write("----------------\n")
        for _, profile in cluster_profiles.iterrows():
            f.write(f"\nCluster {int(profile['Cluster'])}:\n")
            f.write(f"Size: {int(profile['Size'])} customers ({profile['Percentage']:.1f}%)\n")
            
            # Key metrics
            f.write("\nKey Characteristics:\n")
            f.write(f"- Average transaction amount: ₹{profile['avg_transaction_amount_mean']:.2f}\n")
            f.write(f"- Transaction frequency: {profile['transaction_frequency_mean']:.2f} per day\n")
            f.write(f"- Average account balance: ₹{profile['account_balance_mean']:.2f}\n")
            f.write(f"- Average customer age: {profile['age_mean']:.1f} years\n")
            
            # Transaction patterns
            f.write("\nTransaction Patterns:\n")
            f.write(f"- Morning transactions: {profile['morning_transactions_mean']:.1f}\n")
            f.write(f"- Evening transactions: {profile['evening_transactions_mean']:.1f}\n")
            f.write(f"- Weekend transactions: {profile['weekend_transactions_mean']:.1f}\n")
            f.write("\n" + "-"*50 + "\n")

def main():
    """Run the complete analysis pipeline."""
    print("Loading results and data...")
    X, customer_ids, cluster_assignments, scaler, label_encoders = load_results()
    
    print("\nCreating cluster profiles...")
    cluster_profiles = create_cluster_profile(X, cluster_assignments, scaler)
    
    print("\nGenerating visualizations...")
    pca_variance_ratio = visualize_clusters(X, cluster_assignments)
    
    print("\nSaving analysis results...")
    save_analysis_results(cluster_profiles, pca_variance_ratio)
    
    print("\nAnalysis complete! Results saved to:")
    print(f"- Cluster profiles: {os.path.join(OUTPUT_DIR, 'cluster_profiles.csv')}")
    print(f"- Analysis report: {os.path.join(OUTPUT_DIR, 'cluster_analysis_report.txt')}")
    print(f"- Visualizations: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 