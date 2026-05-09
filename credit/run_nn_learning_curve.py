import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library.nn import (
    NeuralNetwork,
    apply_min_max_normalization,
    compute_cross_entropy_cost,
    run_forward_pass,
)


def load_credit_data():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'credit_approval.csv'))
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Numeric columns: indices 1, 2, 7, 13, 14
    # Categorical columns: indices 0, 3, 4, 5, 6, 8, 9, 10, 11, 12
    categorical_cols = {0, 3, 4, 5, 6, 8, 9, 10, 11, 12}
    
    # Encode categorical columns
    X_processed = X.copy().astype(object)
    for col_idx in categorical_cols:
        le = LabelEncoder()
        X_processed[:, col_idx] = le.fit_transform(X[:, col_idx].astype(str))
    
    X_processed = X_processed.astype(float)
    return X_processed, y


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
    print("Loading credit approval dataset...")
    X, y = load_credit_data()
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    rng = np.random.default_rng(42)
    indices = rng.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    # Learning curve sizes - use smaller steps since dataset is smaller than handwriting
    max_size = X.shape[0]
    sizes = list(range(10, max_size + 1, max(1, max_size // 20)))  # ~20 points
    
    if sizes[-1] != max_size:
        sizes.append(max_size)

    costs = []

    model = NeuralNetwork(
        layers=[15, 16, 8, 2],  # 15 input features, binary classification
        regularization=0.01,
        learning_rate=0.01,
        max_iterations=10000,
    )

    for size in sizes:
        print(f"Training with {size} instances...")
        X_subset = X[:size]
        y_subset = y[:size]
        y_one_hot = one_hot_encode(y_subset, num_classes=2)

        model.fit(X_subset, y_one_hot)

        cost = compute_training_cost(model, X_subset, y_one_hot)
        costs.append(cost)
        print(f"  Cost J = {cost:.4f}")

    # Plot learning curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sizes, costs, marker='o', linewidth=2)
    ax.set_xlabel("Number of training instances")
    ax.set_ylabel("Cost J")
    ax.set_title("Neural network learning curve (15-16-8-2, reg=0.01)")
    ax.grid(True, linestyle='--', alpha=0.4)

    out_path = os.path.join(os.path.dirname(__file__), 'nn_learning_curve_15_16_8_2_lambda_0_01.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved learning curve to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
