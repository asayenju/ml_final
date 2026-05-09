import sys
import os
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library.random_forest import RandomForestClassifierScratch
from library.cross_validation import run_stratified_cross_validation

def main():
    print("Loading digits dataset...")
    digits = datasets.load_digits(return_X_y=True)
    X, y = digits[0], digits[1]


    num_features = X.shape[1]
    numeric_cols = set(range(num_features))

    tree_values = [1, 2, 3, 5, 7, 10, 20, 30, 40, 50]
    summary_results = []

    print("Starting evaluations across different n_trees values...")
    print("This may take a few minutes as it trains multiple forests...\n")

    for n in tree_values:
        print(f"=== Evaluating Random Forest with n_trees = {n} ===")
        model = RandomForestClassifierScratch(
            n_trees=n,
            max_depth=15,
            numeric_cols=numeric_cols,
            random_state=42
        )


        res = run_stratified_cross_validation(model, X, y, k=10)


        summary_results.append({
            'n_trees': n,
            'mean_acc': res['mean_accuracy'],
            'std_acc': res['std_accuracy'],
            'mean_f1': res['mean_f1'],
            'std_f1': res['std_f1']
        })
        print(f"Completed n_trees = {n}. Mean Acc: {res['mean_accuracy']:.4f}\n")


    print("\n" + "="*85)
    print(f"{'n_trees':<10} | {'Mean Accuracy':<15} | {'Std Accuracy':<15} | {'Mean F1-Score':<15} | {'Std F1-Score':<15}")
    print("-" * 85)
    for r in summary_results:
        print(f"{r['n_trees']:<10} | {r['mean_acc']:<15.4f} | {r['std_acc']:<15.4f} | {r['mean_f1']:<15.4f} | {r['std_f1']:<15.4f}")
    print("="*85 + "\n")


    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    col_labels = ["n_trees", "Mean Accuracy", "Std Accuracy", "Mean F1-Score", "Std F1-Score"]
    table_data = []
    for r in summary_results:
        table_data.append([
            str(r['n_trees']),
            f"{r['mean_acc']:.4f}",
            f"{r['std_acc']:.4f}",
            f"{r['mean_f1']:.4f}",
            f"{r['std_f1']:.4f}"
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)


    for (i, j), cell in table._cells.items():
        if i == 0:
            cell.set_text_props(weight='bold')

    plt.title("Random Forest 10-Fold CV Results by n_trees", fontsize=14, weight='bold', pad=20)

    out_path = os.path.join(os.path.dirname(__file__), 'rf_ntrees_cv_results.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved results table to {out_path}")


    ntree_values = [r['n_trees'] for r in summary_results]
    mean_acc_values = [r['mean_acc'] for r in summary_results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ntree_values, mean_acc_values, marker='o', linewidth=2)
    ax.set_xlabel("n_trees")
    ax.set_ylabel("Accuracy")
    ax.set_title("Random forest on handwriting dataset: ntree vs accuracy")
    ax.grid(True, linestyle='--', alpha=0.4)

    plot_path = os.path.join(os.path.dirname(__file__), 'rf_ntrees_accuracy.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Saved accuracy plot to {plot_path}")

if __name__ == "__main__":
    main()
