import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# File names
PROCESSED_FEATURES_FILE = 'processed_features.npy'
CUSTOMER_IDS_FILE = 'customer_ids.npy'
SCALER_FILE = 'scaler.joblib'
LABEL_ENCODERS_FILE = 'label_encoders.joblib'
MODEL_FILE = 'customer_segmentation_model.joblib'
CLUSTER_ASSIGNMENTS_FILE = 'cluster_assignments.csv'
CLUSTER_PROFILES_FILE = 'cluster_profiles.txt'

# Column definitions
CATEGORICAL_COLUMNS = ['gender', 'location']  # After aggregation
NUMERICAL_COLUMNS = [
    'transaction_count',
    'avg_transaction_amount',
    'std_transaction_amount',
    'min_transaction_amount',
    'max_transaction_amount',
    'total_transaction_amount',
    'account_balance',
    'age',
    'avg_transaction_time',
    'std_transaction_time',
    'transaction_days_range',
    'transaction_frequency',
    'days_since_last_transaction',
    'morning_transactions',
    'evening_transactions',
    'weekend_transactions'
]

# Model parameters
N_CLUSTERS = 5  # Default number of customer segments
MAX_CLUSTERS = 10  # Maximum number of clusters to try
RANDOM_STATE = 42  # For reproducibility

# Data validation parameters
MIN_AGE = 18
MAX_AGE = 100
MIN_TRANSACTION_AMOUNT = 0
MAX_TRANSACTION_AMOUNT = 1e7  # 10 million
MIN_ACCOUNT_BALANCE = -1e6  # -1 million (allowing some overdraft)
MAX_ACCOUNT_BALANCE = 1e8  # 100 million 