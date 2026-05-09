import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score


def run_stratified_cross_validation(model, X, y, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Ensure X and y are numpy arrays for proper indexing
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
        
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Running fold {fold}/{k}...")
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone model using deepcopy to reset state
        model_clone = copy.deepcopy(model)
        
        # Fit and predict
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        recall_scores.append(recall)
        print(f"  Fold {fold} - Accuracy: {accuracy:.4f}, F1 (macro): {f1:.4f}, Recall (macro): {recall:.4f}")
    
    return {
        'accuracy_scores': accuracy_scores,
        'f1_scores': f1_scores,
        'recall_scores': recall_scores,
        'mean_accuracy': np.mean(accuracy_scores),
        'mean_f1': np.mean(f1_scores),
        'mean_recall': np.mean(recall_scores),
        'std_accuracy': np.std(accuracy_scores),
        'std_f1': np.std(f1_scores),
        'std_recall': np.std(recall_scores)
    }
