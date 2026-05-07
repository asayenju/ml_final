import os
import sys
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Ensure the library module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library.nn import (
    NeuralNetwork,
    apply_min_max_normalization,
    compute_cross_entropy_cost,
    run_forward_pass,
)


def one_hot_encode(labels, num_classes):
    encoded = np.zeros((labels.shape[0], num_classes), dtype=float)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def compute_training_cost(model, features, labels_one_hot):
    if model.normalization_stats is not None:
        features = apply_min_max_normalization(features, model.normalization_stats)
    activations, _ = run_forward_pass(features, model.weight_matrices)
    return compute_cross_entropy_cost(
        labels_one_hot,
        activations[-1],
        model.weight_matrices,
        model.reg_lambda,
    )


def main():
    print("Loading digits dataset...")
    digits = datasets.load_digits(return_X_y=True)
    X, y = digits[0], digits[1]

    rng = np.random.default_rng(42)
    indices = rng.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    # Learning curve sizes
    sizes = range(1, X.shape[0] + 1, 50)  # from 100 to full dataset in steps of 200

    costs = []

    model = NeuralNetwork(
        layers=[64, 16, 10],
        regularization=0.01,
        learning_rate=0.01,
        max_iterations=3000,
    )

    for size in sizes:
        print(f"Training with {size} instances...")
        X_subset = X[:size]
        y_subset = y[:size]
        y_one_hot = one_hot_encode(y_subset, num_classes=10)

        model.fit(X_subset, y_one_hot)

        cost = compute_training_cost(model, X_subset, y_one_hot)
        costs.append(cost)
        print(f"  Cost J = {cost:.4f}")

    # Plot learning curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sizes, costs, marker='o', linewidth=2)
    ax.set_xlabel("Number of training instances")
    ax.set_ylabel("Cost J")
    ax.set_title("Neural network learning curve (64-16-10, reg=0.01)")
    ax.grid(True, linestyle='--', alpha=0.4)

    out_path = os.path.join(os.path.dirname(__file__), 'nn_learning_curve_64_16_10_lambda_0_01.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved learning curve to {out_path}")
    

if __name__ == "__main__":
    main()
