import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library.nn import NeuralNetwork


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


def architecture_label(layers):
    return "-".join(str(size) for size in layers)


def evaluate_nn_arch_reg(X, y, layers, reg_lambda, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = []
    f1_scores = []
    recall_scores = []

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
        recall = recall_score(y_test, y_pred, average='macro')

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        recall_scores.append(recall)

    return {
        "mean_accuracy": float(np.mean(accuracy_scores)),
        "std_accuracy": float(np.std(accuracy_scores)),
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "mean_recall": float(np.mean(recall_scores)),
        "std_recall": float(np.std(recall_scores)),
    }


def evaluate_task(X, y, layers, reg_lambda, k=10):
    """Top-level wrapper for running one architecture+lambda evaluation (picklable)."""
    metrics = evaluate_nn_arch_reg(X, y, layers, reg_lambda, k=k)
    return {
        "architecture": architecture_label(layers),
        "lambda": reg_lambda,
        "mean_acc": metrics["mean_accuracy"],
        "std_acc": metrics["std_accuracy"],
        "mean_recall": metrics["mean_recall"],
        "std_recall": metrics["std_recall"],
        "mean_f1": metrics["mean_f1"],
        "std_f1": metrics["std_f1"],
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

    # Build task list
    tasks = [(layers, reg) for layers in architectures for reg in reg_values]

    # Recommended workers: leave 2 cores free
    try:
        recommended_workers = max(1, min(20, os.cpu_count() - 2))
    except Exception:
        recommended_workers = 4

    # Quick pickling test to ensure ProcessPool can serialize the arguments
    can_parallel = False
    if tasks:
        try:
            pickle.dumps((X, y, tasks[0][0], tasks[0][1]))
            can_parallel = True
        except Exception as e:
            print("Pickle test failed, will run sequentially:", e)

    if can_parallel:
        print(f"Running {len(tasks)} tasks with ProcessPoolExecutor (workers={recommended_workers})\n")
        with ProcessPoolExecutor(max_workers=recommended_workers) as executor:
            future_to_task = {
                executor.submit(evaluate_task, X, y, layers, reg): (layers, reg)
                for (layers, reg) in tasks
            }
            for future in as_completed(future_to_task):
                layers, reg = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"Task {architecture_label(layers)} lambda={reg} raised: {exc}")
                else:
                    results.append(result)
                    print(
                        f"  Done: {result['architecture']} lambda={reg} "
                        f"Acc={result['mean_acc']:.4f} (std {result['std_acc']:.4f}), "
                        f"Recall={result['mean_recall']:.4f}, F1={result['mean_f1']:.4f}"
                    )
    else:
        print("Running sequentially (no parallel execution)\n")
        for layers, reg_lambda in tasks:
            print(f"=== Architecture {architecture_label(layers)} | lambda={reg_lambda} ===")
            result = evaluate_task(X, y, layers, reg_lambda, k=10)
            results.append(result)
            print(
                f"  Done: Acc={result['mean_acc']:.4f} (std {result['std_acc']:.4f}), "
                f"Recall={result['mean_recall']:.4f} (std {result['std_recall']:.4f}), "
                f"F1={result['mean_f1']:.4f} (std {result['std_f1']:.4f})\n"
            )

    # Print results table in the original (architecture, lambda) order
    result_map = {(r['architecture'], r['lambda']): r for r in results}
    ordered_results = []
    for layers, reg in tasks:
        key = (architecture_label(layers), reg)
        if key in result_map:
            ordered_results.append(result_map[key])
        else:
            print(f"Warning: missing result for {key}")

    print("\n" + "=" * 160)
    print(f"{'Architecture':<20} | {'Lambda':<8} | {'Mean Acc':<10} | {'Std Acc':<10} | {'Mean Recall':<12} | {'Std Recall':<12} | {'Mean F1':<10} | {'Std F1':<10}")
    print("-" * 160)
    for r in ordered_results:
        print(f"{r['architecture']:<20} | {r['lambda']:<8} | {r['mean_acc']:<10.4f} | {r['std_acc']:<10.4f} | {r['mean_recall']:<12.4f} | {r['std_recall']:<12.4f} | {r['mean_f1']:<10.4f} | {r['std_f1']:<10.4f}")
    print("=" * 160 + "\n")

    # Report best-performing configurations for each metric
    if results:
        best_acc = max(results, key=lambda r: r['mean_acc'])
        best_recall = max(results, key=lambda r: r['mean_recall'])
        best_f1 = max(results, key=lambda r: r['mean_f1'])

        print("Best configurations:")
        print(f"  Best Mean Accuracy: {best_acc['mean_acc']:.4f} — {best_acc['architecture']} (lambda={best_acc['lambda']})")
        print(f"  Best Mean Recall:   {best_recall['mean_recall']:.4f} — {best_recall['architecture']} (lambda={best_recall['lambda']})")
        print(f"  Best Mean F1:       {best_f1['mean_f1']:.4f} — {best_f1['architecture']} (lambda={best_f1['lambda']})")
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
