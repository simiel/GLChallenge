import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = BASE_DIR / "analysis"

def load_data():
    """Load the transaction dataset."""
    print(f"Loading data from: {DATA_DIR / 'bank_transactions.csv'}")
    df = pd.read_csv(DATA_DIR / "bank_transactions.csv")
    print(f"Loaded {len(df)} transactions")
    return df

def preprocess_data(df):
    """Preprocess the data and create features for churn prediction."""
    print("Preprocessing data...")
    
    # Convert date columns to datetime with mixed format
    print("Converting date columns...")
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'].astype(str), format='mixed', dayfirst=True)
    df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'].astype(str), format='mixed', dayfirst=True)
    
    # Calculate customer age
    print("Calculating customer age...")
    df['CustomerAge'] = (df['TransactionDate'].dt.year - df['CustomerDOB'].dt.year)
    
    # Create customer-level features
    print("Creating customer-level features...")
    customer_features = df.groupby('CustomerID').agg({
        'TransactionAmount (INR)': ['count', 'mean', 'sum', 'std'],
        'CustomerAge': 'first',
        'CustGender': 'first',
        'CustLocation': 'first',
        'CustAccountBalance': 'last',
        'TransactionDate': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = [
        'CustomerID', 'TransactionCount', 'AverageTransactionAmount',
        'TotalTransactionAmount', 'TransactionAmountStd', 'Age', 'Gender',
        'Location', 'AccountBalance', 'FirstTransaction', 'LastTransaction'
    ]
    
    print("Calculating transaction metrics...")
    # Calculate days since last transaction
    max_date = customer_features['LastTransaction'].max()
    customer_features['DaysSinceLastTransaction'] = (
        max_date - customer_features['LastTransaction']
    ).dt.days
    
    # Calculate transaction frequency (days between first and last transaction)
    customer_features['CustomerTenure'] = (
        customer_features['LastTransaction'] - customer_features['FirstTransaction']
    ).dt.days
    
    # Calculate transaction frequency (transactions per month)
    customer_features['TransactionsPerMonth'] = (
        customer_features['TransactionCount'] / 
        (customer_features['CustomerTenure'] / 30)
    ).fillna(0)
    
    print("Defining churn targets...")
    # Define churn based on multiple criteria
    # 1. No transactions in the last 90 days
    # 2. Significant decrease in transaction frequency
    # 3. Account balance below threshold and low transaction frequency
    customer_features['IsChurned'] = (
        (
            (customer_features['DaysSinceLastTransaction'] > 90) &
            (customer_features['TransactionsPerMonth'] < 
             customer_features['TransactionsPerMonth'].median() * 0.5)
        ) |
        (
            (customer_features['AccountBalance'] < 
             customer_features['AccountBalance'].median() * 0.1) &
            (customer_features['TransactionsPerMonth'] < 
             customer_features['TransactionsPerMonth'].median() * 0.3)
        )
    ).astype(int)
    
    print(f"Created features for {len(customer_features)} customers")
    return customer_features

def analyze_churn_factors(df):
    """Analyze factors contributing to churn."""
    print("Analyzing churn factors...")
    churn_analysis = {
        'total_customers': len(df),
        'churned_customers': df['IsChurned'].sum(),
        'churn_rate': df['IsChurned'].mean(),
        'avg_transaction_amount': {
            'churned': df[df['IsChurned'] == 1]['AverageTransactionAmount'].mean(),
            'active': df[df['IsChurned'] == 0]['AverageTransactionAmount'].mean()
        },
        'avg_account_balance': {
            'churned': df[df['IsChurned'] == 1]['AccountBalance'].mean(),
            'active': df[df['IsChurned'] == 0]['AccountBalance'].mean()
        },
        'avg_customer_tenure': {
            'churned': df[df['IsChurned'] == 1]['CustomerTenure'].mean(),
            'active': df[df['IsChurned'] == 0]['CustomerTenure'].mean()
        },
        'avg_transactions_per_month': {
            'churned': df[df['IsChurned'] == 1]['TransactionsPerMonth'].mean(),
            'active': df[df['IsChurned'] == 0]['TransactionsPerMonth'].mean()
        },
        'median_days_since_last_transaction': {
            'churned': df[df['IsChurned'] == 1]['DaysSinceLastTransaction'].median(),
            'active': df[df['IsChurned'] == 0]['DaysSinceLastTransaction'].median()
        }
    }
    
    return churn_analysis

def main():
    """Main function to create churn targets and analyze patterns."""
    print("Starting churn target creation process...")
    df = load_data()
    
    print("\nPreprocessing data and creating churn targets...")
    customer_features = preprocess_data(df)
    
    print("\nAnalyzing churn factors...")
    churn_analysis = analyze_churn_factors(customer_features)
    
    # Save processed dataset
    output_file = DATA_DIR / "customer_features_with_churn.csv"
    print(f"\nSaving processed dataset to: {output_file}")
    customer_features.to_csv(output_file, index=False)
    print("File saved successfully!")
    
    # Print churn analysis
    print("\nChurn Analysis Results:")
    print(f"Total Customers: {churn_analysis['total_customers']}")
    print(f"Churned Customers: {churn_analysis['churned_customers']}")
    print(f"Churn Rate: {churn_analysis['churn_rate']:.2%}")
    print("\nAverage Transaction Amount:")
    print(f"- Churned Customers: ₹{churn_analysis['avg_transaction_amount']['churned']:.2f}")
    print(f"- Active Customers: ₹{churn_analysis['avg_transaction_amount']['active']:.2f}")
    print("\nAverage Account Balance:")
    print(f"- Churned Customers: ₹{churn_analysis['avg_account_balance']['churned']:.2f}")
    print(f"- Active Customers: ₹{churn_analysis['avg_account_balance']['active']:.2f}")
    print("\nAverage Customer Tenure (days):")
    print(f"- Churned Customers: {churn_analysis['avg_customer_tenure']['churned']:.1f}")
    print(f"- Active Customers: {churn_analysis['avg_customer_tenure']['active']:.1f}")
    print("\nAverage Transactions per Month:")
    print(f"- Churned Customers: {churn_analysis['avg_transactions_per_month']['churned']:.2f}")
    print(f"- Active Customers: {churn_analysis['avg_transactions_per_month']['active']:.2f}")
    print("\nMedian Days Since Last Transaction:")
    print(f"- Churned Customers: {churn_analysis['median_days_since_last_transaction']['churned']:.1f}")
    print(f"- Active Customers: {churn_analysis['median_days_since_last_transaction']['active']:.1f}")

if __name__ == "__main__":
    main() 