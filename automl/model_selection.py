from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

MODELS = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

def select_best_model(X, y, cv=5):
    """Select best model without CatBoost dependency"""
    results = {}
    
    for name, model in MODELS.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            results[name] = np.mean(scores)
            print(f"{name:>20}: {results[name]:.4f}")
        except Exception as e:
            print(f"Skipping {name}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No models could be evaluated")
    
    best_model_name = max(results, key=results.get)
    print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")
    return MODELS[best_model_name], results