from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_breast_cancer_data():
    """Load and prepare breast cancer dataset"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target  
    df['diagnosis'] = df['diagnosis'].map({0: 'M', 1: 'B'})  # Convert to M/B labels
    return df

def save_sample_data():
    """Save sample data to CSV"""
    df = load_breast_cancer_data()
    df.to_csv('data/breast_cancer.csv', index=False)

if __name__ == "__main__":
    save_sample_data()