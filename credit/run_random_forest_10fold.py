import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library.random_forest import RandomForestClassifierScratch
from library.cross_validation import run_stratified_cross_validation


def load_credit_data():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'credit_approval.csv'))
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values



    categorical_cols = {0, 3, 4, 5, 6, 8, 9, 10, 11, 12}
    numeric_cols = {1, 2, 7, 13, 14}


    X_processed = X.copy().astype(object)
    for col_idx in categorical_cols:
        le = LabelEncoder()
        X_processed[:, col_idx] = le.fit_transform(X[:, col_idx].astype(str))

    X_processed = X_processed.astype(float)
    return X_processed, y, numeric_cols


def main():
    print("Loading credit approval dataset...")
    X, y, numeric_cols = load_credit_data()

    print(f"Dataset shape: X={X.shape}, y={y.shape}")


    print("Initializing RandomForestClassifierScratch...")
    model = RandomForestClassifierScratch(
        n_trees=15,
        max_depth=15,
        numeric_cols=numeric_cols,
        random_state=42
    )


    print("Starting 10-Fold Stratified Cross Validation...")
    results = run_stratified_cross_validation(model, X, y, k=10)

    print(f"\n--- Cross-Validation Results (10-fold) ---")
    print(f"Mean Accuracy: {results['mean_accuracy'] * 100:.2f}% (std: ±{results['std_accuracy'] * 100:.2f}%)")
    print(f"Mean Recall: {results['mean_recall']:.4f} (std: ±{results['std_recall']:.4f})")
    print(f"Mean F1-Score: {results['mean_f1']:.4f} (std: ±{results['std_f1']:.4f})")


if __name__ == "__main__":
    main()
