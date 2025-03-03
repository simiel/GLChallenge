import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = BASE_DIR / "analysis"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Create timestamp for unique output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_data():
    """Load the customer churn dataset."""
    df = pd.read_csv(DATA_DIR / "mini.csv")
    return df

def generate_basic_stats(df):
    """Generate basic statistics about the dataset."""
    stats = {
        "total_customers": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "summary_stats": df.describe().to_dict()
    }
    
    # Save statistics to JSON
    with open(ANALYSIS_DIR / f"basic_stats_{TIMESTAMP}.json", 'w') as f:
        json.dump(stats, f, indent=4)
    
    return stats

def analyze_churn_distribution(df):
    """Analyze and visualize churn distribution."""
    churn_counts = df['Churn'].value_counts()
    churn_percentages = df['Churn'].value_counts(normalize=True)
    
    # Create pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%')
    plt.title('Customer Churn Distribution')
    plt.savefig(ANALYSIS_DIR / f"churn_distribution_{TIMESTAMP}.png")
    plt.close()
    
    return {
        "churn_counts": churn_counts.to_dict(),
        "churn_percentages": churn_percentages.to_dict()
    }

def analyze_numeric_features(df):
    """Analyze numeric features and their relationship with churn."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_analysis = {}
    
    for col in numeric_cols:
        if col != 'Churn':
            # Create box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Churn', y=col, data=df)
            plt.title(f'{col} Distribution by Churn Status')
            plt.savefig(ANALYSIS_DIR / f"{col}_distribution_{TIMESTAMP}.png")
            plt.close()
            
            # Calculate statistics by churn status
            stats = df.groupby('Churn')[col].describe().to_dict()
            numeric_analysis[col] = stats
    
    # Save numeric analysis to JSON
    with open(ANALYSIS_DIR / f"numeric_analysis_{TIMESTAMP}.json", 'w') as f:
        json.dump(numeric_analysis, f, indent=4)
    
    return numeric_analysis

def analyze_categorical_features(df):
    """Analyze categorical features and their relationship with churn."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_analysis = {}
    
    for col in categorical_cols:
        # Create contingency table
        contingency = pd.crosstab(df[col], df['Churn'])
        
        # Calculate percentages
        percentages = contingency.div(contingency.sum(axis=1), axis=0)
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        percentages.plot(kind='bar', stacked=True)
        plt.title(f'{col} Distribution by Churn Status')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / f"{col}_distribution_{TIMESTAMP}.png")
        plt.close()
        
        categorical_analysis[col] = {
            "contingency_table": contingency.to_dict(),
            "percentages": percentages.to_dict()
        }
    
    # Save categorical analysis to JSON
    with open(ANALYSIS_DIR / f"categorical_analysis_{TIMESTAMP}.json", 'w') as f:
        json.dump(categorical_analysis, f, indent=4)
    
    return categorical_analysis

def generate_correlation_matrix(df):
    """Generate correlation matrix for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f"correlation_matrix_{TIMESTAMP}.png")
    plt.close()
    
    # Save correlation matrix to JSON
    with open(ANALYSIS_DIR / f"correlation_matrix_{TIMESTAMP}.json", 'w') as f:
        json.dump(correlation_matrix.to_dict(), f, indent=4)
    
    return correlation_matrix

def main():
    """Main function to run the analysis."""
    print("Loading data...")
    df = load_data()
    
    print("Generating basic statistics...")
    basic_stats = generate_basic_stats(df)
    
    print("Analyzing churn distribution...")
    churn_analysis = analyze_churn_distribution(df)
    
    print("Analyzing numeric features...")
    numeric_analysis = analyze_numeric_features(df)
    
    print("Analyzing categorical features...")
    categorical_analysis = analyze_categorical_features(df)
    
    print("Generating correlation matrix...")
    correlation_matrix = generate_correlation_matrix(df)
    
    print("Analysis complete! Results saved in the analysis directory.")

if __name__ == "__main__":
    main() 