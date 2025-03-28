from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and generate reports"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save metrics to JSON
    with open('metrics.json', 'w') as f:
        json.dump({'metrics': metrics, 'classification_report': report}, f, indent=2)
    
    return metrics, report