from sklearn.model_selection import RandomizedSearchCV
import numpy as np

PARAM_DISTRIBUTIONS = {
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'HistGradientBoostingClassifier': {
        'max_iter': [50, 100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, None]
    },
    'XGBClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'LGBMClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.01, 0.1]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'penalty': ['l2']
    }
}

def tune_hyperparameters(model, X, y, n_iter=50, cv=3):
    model_name = model.__class__.__name__
    
    if model_name not in PARAM_DISTRIBUTIONS:
        print(f"No tuning parameters for {model_name}, using defaults")
        return model, {}
    
    search = RandomizedSearchCV(
        model,
        PARAM_DISTRIBUTIONS[model_name],
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_