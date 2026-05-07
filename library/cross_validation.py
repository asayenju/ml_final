import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


def run_stratified_cross_validation(model, X, y, k=10):
    """
    Perform stratified k-fold cross-validation on a custom model.
    
    Parameters:
    -----------
    model : object
        A custom model with fit() and predict() methods
    X : array-like
        Feature matrix. Should be a numpy array for advanced indexing.
    y : array-like
        Target labels. Should be a numpy array.
    k : int, default=10
        Number of folds for stratified cross-validation
    
    Returns:
    --------
    dict : A dictionary containing:
        - 'accuracy_scores': list of accuracy scores for each fold
        - 'f1_scores': list of F1 scores (macro average) for each fold
        - 'mean_accuracy': mean accuracy across all folds
        - 'mean_f1': mean F1 score across all folds
        - 'std_accuracy': standard deviation of accuracy
        - 'std_f1': standard deviation of F1 score
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Ensure X and y are numpy arrays for proper indexing
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
        
    accuracy_scores = []
    f1_scores = []
    
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
        
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        print(f"  Fold {fold} - Accuracy: {accuracy:.4f}, F1 (macro): {f1:.4f}")
    
    return {
        'accuracy_scores': accuracy_scores,
        'f1_scores': f1_scores,
        'mean_accuracy': np.mean(accuracy_scores),
        'mean_f1': np.mean(f1_scores),
        'std_accuracy': np.std(accuracy_scores),
        'std_f1': np.std(f1_scores)
    }
