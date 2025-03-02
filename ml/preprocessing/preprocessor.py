import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import sys
import warnings

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class CustomerFeatureProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {col: LabelEncoder() for col in CATEGORICAL_COLUMNS}
        
    def _validate_required_columns(self, df):
        """Validate that all required columns are present in the dataframe."""
        required_columns = [
            'CustomerID', 'CustomerDOB', 'CustGender', 'CustLocation',
            'CustAccountBalance', 'TransactionDate', 'TransactionTime',
            'TransactionAmount (INR)'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _parse_dates(self, df):
        """Parse dates with flexible format handling."""
        print("Parsing dates...")
        df = df.copy()
        
        date_columns = ['CustomerDOB', 'TransactionDate']
        for col in date_columns:
            try:
                # Try different date formats
                df[col] = pd.to_datetime(df[col], format='%d/%m/%y', errors='coerce')
            except ValueError:
                try:
                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                except:
                    print(f"Warning: Some {col} values could not be parsed")
        
        # Replace invalid dates (like 1/1/1800) with NaT
        df.loc[df['CustomerDOB'].dt.year < 1900, 'CustomerDOB'] = pd.NaT
        
        return df
    
    def _calculate_age(self, df):
        """Calculate customer age with proper handling of missing values."""
        print("Calculating customer ages...")
        df = df.copy()
        
        # Calculate age
        df['CustomerAge'] = (df['TransactionDate'] - df['CustomerDOB']).dt.days / 365.25
        
        # Handle invalid ages
        df.loc[df['CustomerAge'] < 0, 'CustomerAge'] = np.nan
        df.loc[df['CustomerAge'] > 100, 'CustomerAge'] = np.nan
        
        # Fill missing ages with median of valid ages
        valid_ages = df['CustomerAge'][(df['CustomerAge'] >= 0) & (df['CustomerAge'] <= 100)]
        median_age = valid_ages.median()
        df['CustomerAge'] = df['CustomerAge'].fillna(median_age)
        
        return df
    
    def _parse_transaction_time(self, df):
        """Parse transaction time with error handling."""
        print("Parsing transaction times...")
        df = df.copy()
        
        # Convert transaction time to string and pad with zeros
        df['TransactionTime'] = df['TransactionTime'].astype(str).str.zfill(6)
        
        try:
            # Extract hours, minutes, seconds
            hours = pd.to_numeric(df['TransactionTime'].str[:2], errors='coerce')
            minutes = pd.to_numeric(df['TransactionTime'].str[2:4], errors='coerce')
            seconds = pd.to_numeric(df['TransactionTime'].str[4:6], errors='coerce')
            
            # Calculate seconds from midnight
            df['SecondsFromMidnight'] = hours * 3600 + minutes * 60 + seconds
            
            # Validate time values
            invalid_times = (hours >= 24) | (minutes >= 60) | (seconds >= 60)
            df.loc[invalid_times, 'SecondsFromMidnight'] = np.nan
            
        except Exception as e:
            print(f"Warning: Error parsing transaction times: {str(e)}")
            df['SecondsFromMidnight'] = np.nan
        
        # Fill missing values with median
        df['SecondsFromMidnight'] = df['SecondsFromMidnight'].fillna(df['SecondsFromMidnight'].median())
        
        return df
    
    def _clean_numeric_values(self, df):
        """Clean numeric columns with proper error handling."""
        print("Cleaning numeric values...")
        df = df.copy()
        
        # Clean monetary values
        numeric_cols = ['CustAccountBalance', 'TransactionAmount (INR)']
        for col in numeric_cols:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove negative values
            df.loc[df[col] < 0, col] = np.nan
            
            # Fill missing values with median
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
        
        return df
    
    def load_and_create_features(self, file_path):
        """Load and create customer behavioral features from transaction data."""
        print(f"Loading data from {file_path}...")
        
        # Read data
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise IOError(f"Error reading file {file_path}: {str(e)}")
        
        # Validate required columns
        self._validate_required_columns(df)
        
        # Process data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = self._clean_numeric_values(df)
            df = self._parse_dates(df)
            df = self._calculate_age(df)
            df = self._parse_transaction_time(df)
        
        # Create customer-level features
        print("Creating customer-level features...")
        customer_features = self._aggregate_customer_features(df)
        
        return customer_features
    
    def _aggregate_customer_features(self, df):
        """Aggregate transaction-level data to customer-level features."""
        # First, create basic aggregations
        customer_features = df.groupby('CustomerID').agg({
            'TransactionAmount (INR)': [
                'count',
                'mean',
                'std',
                'min',
                'max',
                'sum'
            ],
            'CustAccountBalance': 'last',
            'CustomerAge': 'first',
            'CustGender': 'first',
            'CustLocation': 'first',
            'TransactionDate': ['min', 'max'],
            'SecondsFromMidnight': ['mean', 'std']
        })
        
        # Rename columns
        customer_features.columns = [
            'transaction_count', 'avg_transaction_amount', 'std_transaction_amount',
            'min_transaction_amount', 'max_transaction_amount', 'total_transaction_amount',
            'account_balance', 'age', 'gender', 'location',
            'first_transaction_date', 'last_transaction_date',
            'avg_transaction_time', 'std_transaction_time'
        ]
        
        # Reset index to make CustomerID a column
        customer_features = customer_features.reset_index()
        
        # Fill NaN values in std columns with 0
        std_columns = ['std_transaction_amount', 'std_transaction_time']
        customer_features[std_columns] = customer_features[std_columns].fillna(0)
        
        # Calculate additional features
        self._add_derived_features(customer_features, df)
        
        return customer_features
    
    def _add_derived_features(self, customer_features, df):
        """Add derived features to customer features DataFrame."""
        # Calculate transaction days range
        customer_features['transaction_days_range'] = (
            customer_features['last_transaction_date'] - 
            customer_features['first_transaction_date']
        ).dt.days.fillna(0)
        
        # Calculate transaction frequency
        customer_features['transaction_frequency'] = (
            customer_features['transaction_count'] / 
            customer_features['transaction_days_range'].clip(lower=1)
        )
        
        # Calculate recency
        max_date = customer_features['last_transaction_date'].max()
        customer_features['days_since_last_transaction'] = (
            max_date - customer_features['last_transaction_date']
        ).dt.days.fillna(0)
        
        # Calculate time-based features
        time_features = pd.DataFrame({
            'CustomerID': df['CustomerID'],
            'morning': (df['SecondsFromMidnight'] <= 12*3600),
            'evening': (df['SecondsFromMidnight'] > 12*3600),
            'weekend': df['TransactionDate'].dt.dayofweek.isin([5, 6])
        })
        
        # Aggregate time-based features
        time_aggs = time_features.groupby('CustomerID').sum()
        time_aggs.columns = ['morning_transactions', 'evening_transactions', 'weekend_transactions']
        
        # Merge time-based features
        customer_features = customer_features.merge(
            time_aggs, 
            left_on='CustomerID',
            right_index=True,
            how='left'
        )
        
        # Drop date columns
        date_columns = ['first_transaction_date', 'last_transaction_date']
        customer_features.drop(columns=date_columns, inplace=True)
        
        # Fill any remaining NaN values
        customer_features = customer_features.fillna({
            'morning_transactions': 0,
            'evening_transactions': 0,
            'weekend_transactions': 0,
            'transaction_frequency': 0
        })
        
        return customer_features
    
    def preprocess_features(self, df):
        """Preprocess features for clustering."""
        print("Preprocessing features...")
        # Keep CustomerID for reference
        customer_ids = df['CustomerID'].values
        
        # Drop date columns as we've already extracted relevant features from them
        date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        X = df.drop(columns=['CustomerID'] + date_columns)
        
        # Handle categorical columns
        for col in CATEGORICAL_COLUMNS:
            if col in X.columns:
                # Fit and transform
                X[col] = self.label_encoders[col].fit_transform(X[col])
        
        # Scale numerical features
        numerical_columns = [col for col in X.columns if col not in CATEGORICAL_COLUMNS]
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        
        return X.values, customer_ids
    
    def save_preprocessed_data(self, X, customer_ids):
        """Save preprocessed data and preprocessing objects."""
        print(f"Saving preprocessed data to {PROCESSED_DATA_DIR}...")
        
        # Create directories if they don't exist
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Save preprocessed data
        np.save(os.path.join(PROCESSED_DATA_DIR, PROCESSED_FEATURES_FILE), X, allow_pickle=True)
        np.save(os.path.join(PROCESSED_DATA_DIR, CUSTOMER_IDS_FILE), customer_ids, allow_pickle=True)
        
        # Save preprocessing objects
        joblib.dump(self.scaler, os.path.join(PROCESSED_DATA_DIR, SCALER_FILE))
        joblib.dump(self.label_encoders, os.path.join(PROCESSED_DATA_DIR, LABEL_ENCODERS_FILE))
        
        print("Preprocessing complete!")
        return {
            'X_shape': X.shape,
            'n_customers': len(customer_ids)
        }

def run_preprocessing(input_file):
    """Run the complete preprocessing pipeline."""
    processor = CustomerFeatureProcessor()
    
    try:
        # Load and create features
        customer_features = processor.load_and_create_features(input_file)
        
        # Preprocess features
        X, customer_ids = processor.preprocess_features(customer_features)
        
        # Save results
        stats = processor.save_preprocessed_data(X, customer_ids)
        
        return stats
    except Exception as e:
        print(f"\nError in preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess customer transaction data')
    parser.add_argument('--input', '-i', required=True, help='Path to input CSV file')
    
    args = parser.parse_args()
    
    try:
        stats = run_preprocessing(args.input)
        print(f"\nPreprocessing Statistics:")
        print(f"Feature matrix shape: {stats['X_shape']}")
        print(f"Number of customers: {stats['n_customers']}")
    except Exception as e:
        print(f"\nPreprocessing failed: {str(e)}")
        sys.exit(1) 