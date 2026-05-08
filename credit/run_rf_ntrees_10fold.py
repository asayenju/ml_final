import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Ensure the library module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library.random_forest import RandomForestClassifierScratch
from library.cross_validation import run_stratified_cross_validation


def load_credit_data():
    """Load and preprocess credit_approval dataset"""
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'credit_approval.csv'))
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Numeric columns: indices 1, 2, 7, 13, 14
    # Categorical columns: indices 0, 3, 4, 5, 6, 8, 9, 10, 11, 12
    categorical_cols = {0, 3, 4, 5, 6, 8, 9, 10, 11, 12}
    numeric_cols = {1, 2, 7, 13, 14}
    
    # Encode categorical columns
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
    
    tree_values = [1, 2, 3, 5, 7, 10, 20, 30, 40, 50]
    summary_results = []
    
    print("Starting evaluations across different n_trees values...")
    print("This may take a few minutes as it trains multiple forests...\n")
    
    for n in tree_values:
        print(f"=== Evaluating Random Forest with n_trees = {n} ===")
        model = RandomForestClassifierScratch(
            n_trees=n,
            max_depth=15,  # keeping max_depth constant
            numeric_cols=numeric_cols,
            random_state=42
        )
        
        # Run 10-fold cross validation
        res = run_stratified_cross_validation(model, X, y, k=10)
        
        # Store results for the final table
        summary_results.append({
            'n_trees': n,
            'mean_acc': res['mean_accuracy'],
            'std_acc': res['std_accuracy'],
            'mean_recall': res['mean_recall'],
            'std_recall': res['std_recall'],
            'mean_f1': res['mean_f1'],
            'std_f1': res['std_f1']
        })
        print(f"Completed n_trees = {n}. Mean Acc: {res['mean_accuracy']:.4f}\n")
        
    # Print the final formatted table
    print("\n" + "="*115)
    print(f"{'n_trees':<10} | {'Mean Accuracy':<15} | {'Std Accuracy':<15} | {'Mean Recall':<15} | {'Std Recall':<15} | {'Mean F1-Score':<15} | {'Std F1-Score':<15}")
    print("-" * 115)
    for r in summary_results:
        print(f"{r['n_trees']:<10} | {r['mean_acc']:<15.4f} | {r['std_acc']:<15.4f} | {r['mean_recall']:<15.4f} | {r['std_recall']:<15.4f} | {r['mean_f1']:<15.4f} | {r['std_f1']:<15.4f}")
    print("="*115 + "\n")

    # Generate and save matplotlib table
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    col_labels = ["n_trees", "Mean Accuracy", "Std Accuracy", "Mean Recall", "Std Recall", "Mean F1-Score", "Std F1-Score"]
    table_data = []
    for r in summary_results:
        table_data.append([
            str(r['n_trees']), 
            f"{r['mean_acc']:.4f}", 
            f"{r['std_acc']:.4f}",
            f"{r['mean_recall']:.4f}",
            f"{r['std_recall']:.4f}",
            f"{r['mean_f1']:.4f}", 
            f"{r['std_f1']:.4f}"
        ])
        
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    # Make header bold
    for (i, j), cell in table._cells.items():
        if i == 0:
            cell.set_text_props(weight='bold')
    
    plt.title("Random Forest 10-Fold CV Results by n_trees", fontsize=14, weight='bold', pad=20)
    
    out_path = os.path.join(os.path.dirname(__file__), 'rf_ntrees_cv_results.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved results table to {out_path}")
    plt.close()

    # Plot accuracy vs n_trees
    ntree_values = [r['n_trees'] for r in summary_results]
    mean_acc_values = [r['mean_acc'] for r in summary_results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ntree_values, mean_acc_values, marker='o', linewidth=2)
    ax.set_xlabel("n_trees")
    ax.set_ylabel("Accuracy")
    ax.set_title("Random forest on credit approval dataset: ntree vs accuracy")
    ax.grid(True, linestyle='--', alpha=0.4)

    out_path = os.path.join(os.path.dirname(__file__), 'rf_ntrees_accuracy_plot.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved accuracy plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
