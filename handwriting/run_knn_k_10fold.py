import os
import sys
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Ensure the library module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library.knn import KNNClassifier
from library.cross_validation import run_stratified_cross_validation


def main():
    print("Loading digits dataset...")
    digits = datasets.load_digits(return_X_y=True)
    X, y = digits[0], digits[1]

    k_values = range(1, 51, 2)  
    summary_results = []

    print("Starting KNN evaluations across different k values...\n")

    for k in k_values:
        print(f"=== Evaluating KNN with k = {k} ===")
        model = KNNClassifier(k=k)
        res = run_stratified_cross_validation(model, X, y, k=10)

        summary_results.append({
            "k": k,
            "mean_acc": res["mean_accuracy"],
            "std_acc": res["std_accuracy"],
            "mean_f1": res["mean_f1"],
            "std_f1": res["std_f1"],
        })
        print(f"Completed k = {k}. Mean Acc: {res['mean_accuracy']:.4f}\n")

    # Print the final formatted table
    print("\n" + "=" * 80)
    print(f"{'k':<6} | {'Mean Accuracy':<15} | {'Std Accuracy':<15} | {'Mean F1-Score':<15} | {'Std F1-Score':<15}")
    print("-" * 80)
    for r in summary_results:
        print(f"{r['k']:<6} | {r['mean_acc']:<15.4f} | {r['std_acc']:<15.4f} | {r['mean_f1']:<15.4f} | {r['std_f1']:<15.4f}")
    print("=" * 80 + "\n")

    # Plot accuracy vs k
    k_axis = [r["k"] for r in summary_results]
    acc_axis = [r["mean_acc"] for r in summary_results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_axis, acc_axis, marker='o', linewidth=2)
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_title("KNN on handwriting dataset: k vs accuracy")
    ax.grid(True, linestyle='--', alpha=0.4)

    out_path = os.path.join(os.path.dirname(__file__), 'knn_k_accuracy.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved accuracy plot to {out_path}")


if __name__ == "__main__":
    main()
