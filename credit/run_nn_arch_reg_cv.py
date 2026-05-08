import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Ensure the library module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library.nn import NeuralNetwork


def load_credit_data():
    """Load and preprocess credit_approval dataset"""
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


def architecture_label(layers):
    return "-".join(str(size) for size in layers)


def evaluate_nn_arch_reg(X, y, layers, reg_lambda, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = []
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  Fold {fold}/{k}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        y_train_one_hot = one_hot_encode(y_train, num_classes=2)

        model = NeuralNetwork(
            layers=layers,
            regularization=reg_lambda,
            learning_rate=0.1,
            max_iterations=3000,
        )
        model.fit(X_train, y_train_one_hot)
        probs = model.predict(X_test)
        y_pred = np.argmax(probs, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)

    return {
        "mean_accuracy": float(np.mean(accuracy_scores)),
        "std_accuracy": float(np.std(accuracy_scores)),
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
    }


def main():
    print("Loading credit approval dataset...")
    X, y = load_credit_data()
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Architectures: input=15, output=2 (binary classification)
    architectures = [
        [15, 2, 2],
        [15, 4, 2],
        [15, 8, 2],
        [15, 16, 2],
        [15, 16, 8, 2],
        [15, 16, 8, 4, 2],
        [15, 16, 8, 4, 2, 2],
    ]
    reg_values = [0, 0.01, 0.1, 1, 10]

    results = []

    print("Starting NN evaluations across architectures and regularization values...\n")
    for layers in architectures:
        arch_name = architecture_label(layers)
        for reg_lambda in reg_values:
            print(f"=== Architecture {arch_name} | lambda={reg_lambda} ===")
            metrics = evaluate_nn_arch_reg(X, y, layers, reg_lambda, k=10)
            results.append({
                "architecture": arch_name,
                "lambda": reg_lambda,
                "mean_acc": metrics["mean_accuracy"],
                "std_acc": metrics["std_accuracy"],
                "mean_f1": metrics["mean_f1"],
                "std_f1": metrics["std_f1"],
            })
            print(
                f"  Done: Acc={metrics['mean_accuracy']:.4f} (std {metrics['std_accuracy']:.4f}), "
                f"F1={metrics['mean_f1']:.4f} (std {metrics['std_f1']:.4f})\n"
            )

    # Print results table
    print("\n" + "=" * 130)
    print(f"{'Architecture':<20} | {'Lambda':<8} | {'Mean Acc':<10} | {'Std Acc':<10} | {'Mean F1':<10} | {'Std F1':<10}")
    print("-" * 130)
    for r in results:
        print(f"{r['architecture']:<20} | {r['lambda']:<8} | {r['mean_acc']:<10.4f} | {r['std_acc']:<10.4f} | {r['mean_f1']:<10.4f} | {r['std_f1']:<10.4f}")
    print("=" * 130 + "\n")


if __name__ == "__main__":
    main()
