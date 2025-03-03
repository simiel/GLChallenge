import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

# Set up paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

def load_model_artifacts():
    """Load the trained model and its artifacts."""
    print("Loading model artifacts...")
    
    # Load the model
    model = joblib.load(MODELS_DIR / "churn_model.joblib")
    print("Model loaded successfully")
    
    # Load the scaler
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    print("Scaler loaded successfully")
    
    # Load feature names
    with open(MODELS_DIR / "feature_names.json", 'r') as f:
        feature_names = json.load(f)
    print("Feature names loaded successfully")
    
    # Load label encoders
    label_encoders = joblib.load(MODELS_DIR / "label_encoders.joblib")
    print("Label encoders loaded successfully")
    
    # Print available categories
    print("\nAvailable categories:")
    for col, encoder in label_encoders.items():
        print(f"{col}: {encoder.classes_.tolist()}")
    
    return model, scaler, feature_names, label_encoders

def create_sample_data(label_encoders):
    """Create sample customer data for prediction."""
    # Get available categories
    available_genders = label_encoders['Gender'].classes_.tolist()
    available_locations = label_encoders['Location'].classes_.tolist()
    
    # Create sample customers with different profiles
    samples = [
        {
            # High-value active customer
            'TransactionCount': 50,
            'AverageTransactionAmount': 5000,
            'TotalTransactionAmount': 250000,
            'TransactionAmountStd': 1000,
            'Age': 35,
            'AccountBalance': 100000,
            'DaysSinceLastTransaction': 5,
            'CustomerTenure': 365,
            'TransactionsPerMonth': 15,
            'Gender': available_genders[0],
            'Location': available_locations[0]
        },
        {
            # Potentially churning customer
            'TransactionCount': 5,
            'AverageTransactionAmount': 1000,
            'TotalTransactionAmount': 5000,
            'TransactionAmountStd': 200,
            'Age': 45,
            'AccountBalance': 1000,
            'DaysSinceLastTransaction': 60,
            'CustomerTenure': 180,
            'TransactionsPerMonth': 1,
            'Gender': available_genders[-1],
            'Location': available_locations[-1]
        },
        {
            # New customer
            'TransactionCount': 10,
            'AverageTransactionAmount': 3000,
            'TotalTransactionAmount': 30000,
            'TransactionAmountStd': 500,
            'Age': 28,
            'AccountBalance': 50000,
            'DaysSinceLastTransaction': 15,
            'CustomerTenure': 30,
            'TransactionsPerMonth': 10,
            'Gender': available_genders[0],
            'Location': available_locations[1] if len(available_locations) > 1 else available_locations[0]
        }
    ]
    
    return pd.DataFrame(samples)

def prepare_features(df, feature_names, label_encoders):
    """Prepare features for prediction."""
    # Create a copy of the dataframe
    df_prepared = df.copy()
    
    # Encode categorical features
    for col, encoder in label_encoders.items():
        df_prepared[col + '_encoded'] = encoder.transform(df[col])
    
    # Select only the features used by the model
    X = df_prepared[feature_names]
    
    return X

def predict_churn(model, scaler, X):
    """Make churn predictions."""
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Get predictions and probabilities
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    return predictions, probabilities

def main():
    """Main function to demonstrate churn prediction."""
    # Load model artifacts
    model, scaler, feature_names, label_encoders = load_model_artifacts()
    
    # Create sample data
    print("\nCreating sample customer data...")
    sample_data = create_sample_data(label_encoders)
    
    # Prepare features
    print("\nPreparing features...")
    X = prepare_features(sample_data, feature_names, label_encoders)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions, probabilities = predict_churn(model, scaler, X)
    
    # Create results dataframe
    results = pd.DataFrame({
        'CustomerProfile': ['High-value Active', 'Potentially Churning', 'New Customer'],
        'Gender': sample_data['Gender'],
        'Location': sample_data['Location'],
        'AccountBalance': sample_data['AccountBalance'],
        'TransactionsPerMonth': sample_data['TransactionsPerMonth'],
        'DaysSinceLastTransaction': sample_data['DaysSinceLastTransaction'],
        'PredictedChurn': predictions,
        'ChurnProbability': probabilities
    })
    
    # Display results
    print("\nPrediction Results:")
    print("\nDetailed Results:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results.to_string(index=False))
    
    # Print interpretation
    print("\nInterpretation:")
    for i, row in results.iterrows():
        print(f"\nCustomer Profile: {row['CustomerProfile']}")
        print(f"- Gender: {row['Gender']}")
        print(f"- Location: {row['Location']}")
        print(f"- Account Balance: ₹{row['AccountBalance']:,.2f}")
        print(f"- Transactions per Month: {row['TransactionsPerMonth']:.1f}")
        print(f"- Days Since Last Transaction: {row['DaysSinceLastTransaction']:.0f}")
        print(f"- Churn Probability: {row['ChurnProbability']:.1%}")
        print(f"- Prediction: {'High Risk of Churn' if row['PredictedChurn'] else 'Likely to Stay'}")

if __name__ == "__main__":
    main() 