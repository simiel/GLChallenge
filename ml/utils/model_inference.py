import joblib
import pandas as pd
import numpy as np
from datetime import datetime

def load_model(model_path='../models/customer_segmentation.joblib'):
    """Load the trained clustering model."""
    return joblib.load(model_path)

def prepare_input(customer_data, scaler, label_encoders):
    """Prepare input data for clustering."""
    # Convert to DataFrame if dict
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    
    # Handle categorical variables
    categorical_columns = ['gender', 'location']
    for column in categorical_columns:
        if column in customer_data and column in label_encoders:
            customer_data[column] = label_encoders[column].transform(customer_data[column])
    
    # Scale numerical features
    numerical_columns = [
        'transaction_count', 'avg_transaction_amount', 'std_transaction_amount',
        'min_transaction_amount', 'max_transaction_amount', 'total_transaction_amount',
        'account_balance', 'age', 'days_between_transactions', 'transaction_frequency',
        'days_since_last_transaction', 'morning_transactions', 'evening_transactions',
        'weekend_transactions'
    ]
    
    # Ensure all numerical columns exist
    for col in numerical_columns:
        if col not in customer_data:
            raise ValueError(f"Missing required feature: {col}")
    
    customer_data[numerical_columns] = scaler.transform(customer_data[numerical_columns])
    
    return customer_data

def get_cluster_profile(cluster_id):
    """Get the profile description for a given cluster."""
    # Load cluster profiles
    try:
        with open('../outputs/cluster_profiles.txt', 'r') as f:
            profiles = f.read()
        
        # Find the relevant cluster section
        cluster_sections = profiles.split('\n\n')
        for section in cluster_sections:
            if section.startswith(f"Cluster {cluster_id}:"):
                return section
        
        return "Cluster profile not found."
    except FileNotFoundError:
        return "Cluster profiles file not found."

def assign_customer_segment(customer_data, model, scaler, label_encoders):
    """Assign a customer to a segment and provide segment characteristics."""
    # Prepare input data
    X = prepare_input(customer_data, scaler, label_encoders)
    
    # Get cluster assignment
    cluster_id = model.predict(X)[0]
    
    # Get cluster profile
    cluster_profile = get_cluster_profile(cluster_id)
    
    # Calculate distance to cluster center
    distance_to_center = np.linalg.norm(X - model.cluster_centers_[cluster_id])
    
    return {
        'cluster_id': int(cluster_id),
        'cluster_profile': cluster_profile,
        'confidence_score': 1.0 / (1.0 + distance_to_center),  # Convert distance to similarity score
        'distance_to_center': float(distance_to_center)
    }

if __name__ == "__main__":
    # Example usage
    model = load_model()
    
    # Example customer data
    customer_data = {
        'transaction_count': 10,
        'avg_transaction_amount': 5000.0,
        'std_transaction_amount': 1000.0,
        'min_transaction_amount': 1000.0,
        'max_transaction_amount': 10000.0,
        'total_transaction_amount': 50000.0,
        'account_balance': 100000.0,
        'age': 35,
        'gender': 'M',
        'location': 'MUMBAI',
        'days_between_transactions': 30,
        'transaction_frequency': 0.33,
        'days_since_last_transaction': 15,
        'morning_transactions': 4,
        'evening_transactions': 6,
        'weekend_transactions': 3
    }
    
    # Note: In practice, you would need to load the saved scaler and label_encoders
    try:
        result = assign_customer_segment(customer_data, model, scaler, label_encoders)
        print("\nCustomer Segmentation Results:")
        print(f"Assigned Cluster: {result['cluster_id']}")
        print(f"\nCluster Profile:")
        print(result['cluster_profile'])
        print(f"\nConfidence Score: {result['confidence_score']:.2%}")
    except Exception as e:
        print(f"Error assigning segment: {str(e)}") 