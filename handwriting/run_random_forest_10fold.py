import sys
import os
import numpy as np
from sklearn import datasets

# Ensure the library module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library.random_forest import RandomForestClassifierScratch
from library.cross_validation import run_stratified_cross_validation


def main():
    print("Loading digits dataset...")
    digits = datasets.load_digits(return_X_y=True)
    X, y = digits[0], digits[1]
    
    # In this dataset, all 64 attributes are numeric (pixel values)
    num_features = X.shape[1]
    numeric_cols = set(range(num_features))
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Create the model instance
    print("Initializing RandomForestClassifierScratch...")
    model = RandomForestClassifierScratch(
        n_trees=15,
        max_depth=15,
        numeric_cols=numeric_cols,
        random_state=42
    )
    
    # Run 10-fold cross validation
    print("Starting 10-Fold Stratified Cross Validation...")
    results = run_stratified_cross_validation(model, X, y, k=10)
    
    print(f"\n--- Cross-Validation Results (10-fold) ---")
    print(f"Mean Accuracy: {results['mean_accuracy'] * 100:.2f}% (std: ±{results['std_accuracy'] * 100:.2f}%)")
    print(f"Mean F1-Score: {results['mean_f1']:.4f} (std: ±{results['std_f1']:.4f})")

if __name__ == "__main__":
    main()
