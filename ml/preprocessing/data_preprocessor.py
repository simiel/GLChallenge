import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_create_features(file_path):
    """Load and create customer behavioral features from transaction data."""
    # Read the data
    df = pd.read_csv(file_path)
    
    # Convert dates to datetime
    df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], format='%d/%m/%y')
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='%d/%m/%y')
    
    # Calculate age at transaction
    df['CustomerAge'] = (df['TransactionDate'] - df['CustomerDOB']).dt.days / 365.25
    
    # Convert transaction time to seconds since midnight
    df['TransactionTime'] = pd.to_numeric(df['TransactionTime'].astype(str).str.zfill(6))
    df['SecondsFromMidnight'] = (df['TransactionTime'] // 10000 * 3600 + 
                                (df['TransactionTime'] % 10000) // 100 * 60 + 
                                df['TransactionTime'] % 100)
    
    # Clean account balance (handle missing values)
    df['CustAccountBalance'] = pd.to_numeric(df['CustAccountBalance'], errors='coerce')
    df['CustAccountBalance'].fillna(df['CustAccountBalance'].median(), inplace=True)
    
    # Group by CustomerID to create customer-level features
    customer_features = df.groupby('CustomerID').agg({
        'TransactionAmount (INR)': [
            ('transaction_count', 'count'),
            ('avg_transaction_amount', 'mean'),
            ('std_transaction_amount', lambda x: x.std() if len(x) > 1 else 0),
            ('min_transaction_amount', 'min'),
            ('max_transaction_amount', 'max'),
            ('total_transaction_amount', 'sum')
        ],
        'CustAccountBalance': ('account_balance', 'last'),
        'CustomerAge': ('age', 'first'),
        'CustGender': ('gender', 'first'),
        'CustLocation': ('location', 'first'),
        'TransactionDate': [
            ('first_transaction_date', 'min'),
            ('last_transaction_date', 'max'),
            ('transaction_days_range', lambda x: (x.max() - x.min()).days)
        ],
        'SecondsFromMidnight': [
            ('avg_transaction_time', 'mean'),
            ('std_transaction_time', lambda x: x.std() if len(x) > 1 else 0)
        ]
    })
    
    # Flatten column names
    customer_features.columns = customer_features.columns.map('_'.join)
    customer_features = customer_features.reset_index()
    
    # Calculate additional behavioral features
    customer_features['transaction_frequency'] = (
        customer_features['TransactionAmount (INR)_transaction_count'] / 
        customer_features['TransactionDate_transaction_days_range'].clip(lower=1)
    )
    
    # Calculate recency (days since last transaction)
    max_date = customer_features['TransactionDate_last_transaction_date'].max()
    customer_features['days_since_last_transaction'] = (
        max_date - customer_features['TransactionDate_last_transaction_date']
    ).dt.days
    
    # Calculate time-based features
    customer_features['morning_transactions'] = df[df['SecondsFromMidnight'] <= 12*3600].groupby('CustomerID').size()
    customer_features['evening_transactions'] = df[df['SecondsFromMidnight'] > 12*3600].groupby('CustomerID').size()
    customer_features['weekend_transactions'] = df[df['TransactionDate'].dt.dayofweek.isin([5,6])].groupby('CustomerID').size()
    
    # Fill NaN values with 0 for transaction counts
    customer_features = customer_features.fillna({
        'morning_transactions': 0,
        'evening_transactions': 0,
        'weekend_transactions': 0
    })
    
    # Drop date columns and CustomerID as they're no longer needed
    date_columns = [col for col in customer_features.columns if 'date' in col.lower()]
    customer_features.drop(date_columns, axis=1, inplace=True)
    
    return customer_features

def preprocess_features(df):
    """Preprocess features for clustering."""
    # Keep CustomerID for reference but don't include in clustering
    customer_ids = df['CustomerID']
    X = df.drop('CustomerID', axis=1)
    
    # Handle categorical variables
    categorical_columns = ['gender', 'location']
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    
    # Scale numerical features
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    return X, customer_ids, scaler, label_encoders

def save_preprocessed_data(X, customer_ids, output_dir='../data/processed'):
    """Save preprocessed data to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the preprocessed data
    np.save(f'{output_dir}/X_processed.npy', X)
    np.save(f'{output_dir}/customer_ids.npy', customer_ids)

if __name__ == "__main__":
    # Load and create features
    print("Loading data and creating features...")
    customer_features = load_and_create_features('../data/raw/mini.csv')
    
    # Preprocess features
    print("Preprocessing features...")
    X, customer_ids, scaler, label_encoders = preprocess_features(customer_features)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    save_preprocessed_data(X, customer_ids)
    
    print("Feature creation complete. Shape of feature matrix:", X.shape) 