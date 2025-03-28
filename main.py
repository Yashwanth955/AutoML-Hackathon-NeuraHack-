import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from automl import (
    preprocess_data,
    select_best_model,
    tune_hyperparameters,
    evaluate_model
)
from data.load_dataset import load_breast_cancer_data
import sys

def load_config():
    try:
        with open('config/params.yaml') as f:
            config = yaml.safe_load(f)
            assert 'target_column' in config, "Missing target_column in config"
            return config
    except Exception as e:
        print(f"Config Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    print("=== AutoML Pipeline ===")
    
    # 1. Config
    print("\n[1/4] Loading config...")
    config = load_config()
    print(f"Target: {config['target_column']}")
    
    # 2. Data
    print("\n[2/4] Loading data...")
    try:
        data = load_breast_cancer_data()
        print(f"Data shape: {data.shape}")
        print("Target counts:")
        print(data[config['target_column']].value_counts())
    except Exception as e:
        print(f"Data Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # 3. Preprocessing
    print("\n[3/4] Preprocessing...")
    X = data.drop(columns=[config['target_column']])
    y = data[config['target_column']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.get('test_size', 0.2),
        random_state=42,
        stratify=y
    )
    
    # 4. Modeling
    print("\n[4/4] Running pipeline...")
    try:
        model, scores = select_best_model(X_train, y_train)
        print("\nModel Scores:")
        for name, score in scores.items():
            print(f"{name:>20}: {score:.4f}")
    except Exception as e:
        print(f"Model Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()