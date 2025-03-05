import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
from datetime import datetime

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Create timestamp for unique output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_data():
    """Load the processed customer dataset with churn targets."""
    print("Loading processed data...")
    # First, read the CSV to get column names
    df = pd.read_csv(DATA_DIR / "customer_features_with_churn.csv", nrows=0)
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create dtype dictionary for numeric columns
    dtypes = {col: 'float32' for col in numeric_columns}
    dtypes['IsChurned'] = 'int8'  # Always keep IsChurned as int8
    
    # Now read the full CSV with appropriate dtypes
    df = pd.read_csv(DATA_DIR / "customer_features_with_churn.csv", dtype=dtypes)
    print(f"Loaded {len(df)} customer records")
    return df

def clean_data(df):
    """Clean the data by handling infinite values and outliers."""
    print("\nCleaning data...")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['float32', 'float64']).columns
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For each numeric column
    for col in numeric_cols:
        if col != 'IsChurned':
            # Calculate statistics
            median = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds for outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Replace outliers and NaN with median
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            df[col] = df[col].fillna(median)
            
            print(f"Cleaned column {col}: {df[col].isna().sum()} NaN values remaining")
    
    return df

def prepare_features(df):
    """Prepare features for model training."""
    print("\nPreparing features...")
    
    # Select numeric features that exist in the dataframe
    numeric_features = [
        'TransactionCount', 'AverageTransactionAmount', 'TotalTransactionAmount',
        'TransactionAmountStd', 'Age', 'AccountBalance', 'DaysSinceLastTransaction',
        'CustomerTenure', 'TransactionsPerMonth'
    ]
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    # Handle categorical features efficiently
    print("Encoding categorical features...")
    label_encoders = {}
    categorical_features = ['Gender', 'Location']
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    for cat_col in categorical_features:
        label_encoders[cat_col] = LabelEncoder()
        df[cat_col + '_encoded'] = label_encoders[cat_col].fit_transform(df[cat_col])
    
    # Save label encoders
    joblib.dump(label_encoders, MODELS_DIR / "label_encoders.joblib")
    
    # Combine features
    feature_cols = numeric_features + [f + '_encoded' for f in categorical_features]
    X = df[feature_cols].copy()
    y = df['IsChurned']
    
    print(f"Prepared {len(feature_cols)} features")
    
    # Save feature names
    with open(MODELS_DIR / "feature_names.json", 'w') as f:
        json.dump(feature_cols, f)
    
    return X, y

def train_model(X, y):
    """Train and evaluate the churn prediction model."""
    print("\nTraining model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use RobustScaler instead of StandardScaler to handle outliers better
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    
    # Train Random Forest (memory efficient configuration)
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results = {
        'random_forest': {
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    }
    
    print(f"\nRandom Forest ROC AUC Score: {roc_auc:.4f}")
    
    # Save results
    with open(ARTIFACTS_DIR / f"model_evaluation_{TIMESTAMP}.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv(ARTIFACTS_DIR / f"feature_importance_{TIMESTAMP}.csv", index=False)
    
    return rf_model, scaler

def save_model(model, scaler):
    """Save the trained model and scaler."""
    print("\nSaving model artifacts...")
    
    # Save the model
    model_path = MODELS_DIR / "churn_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save the scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

def main():
    """Main function to train and save the churn prediction model."""
    # Load data
    df = load_data()
    
    # Clean data
    df = clean_data(df)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train model
    best_model, scaler = train_model(X, y)
    
    # Save model artifacts
    save_model(best_model, scaler)
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main() 