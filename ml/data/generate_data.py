import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def generate_customer_data(n_customers=1000, n_transactions_per_customer_range=(5, 20)):
    """Generate synthetic customer transaction data."""
    # Set random seed for reproducibility
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    
    # Generate customer base data
    customers = []
    for i in range(n_customers):
        customer_id = f'C{str(i+1).zfill(7)}'
        
        # Generate date of birth (between 1960 and 2000)
        dob = datetime(1960, 1, 1) + timedelta(
            days=random.randint(0, (datetime(2000, 12, 31) - datetime(1960, 1, 1)).days)
        )
        
        # Generate other customer attributes
        gender = random.choice(['M', 'F'])
        location = random.choice([
            'MUMBAI', 'DELHI', 'BANGALORE', 'CHENNAI', 'KOLKATA',
            'HYDERABAD', 'PUNE', 'AHMEDABAD', 'JAIPUR', 'LUCKNOW'
        ])
        
        # Generate account balance (log-normal distribution)
        account_balance = np.random.lognormal(mean=10, sigma=1)
        
        # Generate transactions for this customer
        n_transactions = random.randint(*n_transactions_per_customer_range)
        
        # Generate transaction dates within last 2 years
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=730)  # 2 years
        
        for _ in range(n_transactions):
            # Generate transaction date and time
            trans_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )
            trans_time = f"{random.randint(0, 23):02d}{random.randint(0, 59):02d}{random.randint(0, 59):02d}"
            
            # Generate transaction amount (log-normal distribution)
            trans_amount = np.random.lognormal(mean=7, sigma=1.5)  # Mean ~₹1100, with high variance
            
            # Create transaction record
            customers.append({
                'TransactionID': f'T{len(customers)+1}',
                'CustomerID': customer_id,
                'CustomerDOB': dob.strftime('%d/%m/%y'),
                'CustGender': gender,
                'CustLocation': location,
                'CustAccountBalance': round(account_balance, 2),
                'TransactionDate': trans_date.strftime('%d/%m/%y'),
                'TransactionTime': trans_time,
                'TransactionAmount (INR)': round(trans_amount, 2)
            })
    
    # Create DataFrame
    df = pd.DataFrame(customers)
    
    # Sort by TransactionDate and TransactionTime
    df['temp_date'] = pd.to_datetime(df['TransactionDate'], format='%d/%m/%y')
    df = df.sort_values(['temp_date', 'TransactionTime']).drop('temp_date', axis=1)
    
    return df

def save_data(df, output_path):
    """Save generated data to CSV file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated data saved to: {output_path}")
    print(f"Number of customers: {df['CustomerID'].nunique()}")
    print(f"Number of transactions: {len(df)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic customer transaction data')
    parser.add_argument('--customers', '-c', type=int, default=1000,
                      help='Number of customers to generate')
    parser.add_argument('--min-trans', type=int, default=5,
                      help='Minimum transactions per customer')
    parser.add_argument('--max-trans', type=int, default=20,
                      help='Maximum transactions per customer')
    parser.add_argument('--output', '-o', default=os.path.join(RAW_DATA_DIR, 'synthetic_data.csv'),
                      help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Generate data
    print(f"Generating data for {args.customers} customers...")
    df = generate_customer_data(
        n_customers=args.customers,
        n_transactions_per_customer_range=(args.min_trans, args.max_trans)
    )
    
    # Save data
    save_data(df, args.output) 