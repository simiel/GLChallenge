# Churn Prediction Pipeline

This module implements an end-to-end customer churn prediction pipeline using both unsupervised and supervised learning techniques.

## Overview

The pipeline consists of the following steps:
1. Data preprocessing and feature engineering
2. Unsupervised learning to generate initial churn labels
3. Supervised learning for the final churn prediction model
4. Model evaluation and artifact storage
5. Prediction pipeline for new data

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model using your data:

```bash
python main.py --mode train --data_path path/to/your/data.csv
```

Optional arguments:
- `--model_dir`: Directory to save model artifacts (default: 'models')
- `--random_state`: Random seed for reproducibility (default: 42)

### Making Predictions

To make predictions on new data:

```bash
python main.py --mode predict --data_path path/to/new/data.csv
```

## Input Data Format

The input data should be a CSV file with the following characteristics:
- Numerical columns will be automatically detected and preprocessed
- Categorical columns will be automatically detected and one-hot encoded
- No target column is needed for training (unsupervised approach)

## Output

### Training Mode
- Trained model and preprocessing artifacts saved to the specified model directory
- Training metrics printed to console
- Feature importance analysis

### Prediction Mode
- Predictions and probabilities for each customer
- Summary statistics of predictions
- Churn distribution in the dataset

## Model Details

### Unsupervised Learning
- Uses an ensemble of KMeans and Gaussian Mixture Models
- Automatically identifies the churn cluster based on behavior patterns
- Combines predictions from multiple clustering approaches

### Supervised Learning
- Random Forest Classifier (default)
- Option to use Gradient Boosting
- Feature importance analysis
- Comprehensive evaluation metrics

## Files
- `main.py`: Main script for running the pipeline
- `data_preprocessing.py`: Data preprocessing and feature engineering
- `unsupervised_churn.py`: Unsupervised learning for churn labeling
- `supervised_model.py`: Supervised learning and model evaluation
- `requirements.txt`: Required Python packages

## Example

```python
from main import train_pipeline, predict_pipeline

# Train the model
artifacts = train_pipeline('data/training_data.csv')

# Make predictions
results, _ = predict_pipeline('data/new_data.csv')
print(results.head())
``` 