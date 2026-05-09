"""
Neural Network learning curve for the Oxford Parkinson's Disease dataset.

Uses the best architecture found in run_nn_10fold.py.
Trains the NN on increasing training-set sizes and records the final training
cost J at each size (averaged over 5 random seeds for stability).

Saves the learning curve plot to the writeup directory.
"""
import csv
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
WRITEUP_DIR = ROOT.parent.parent / "ml-project-writeup" / "parkinsons_dataset"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.nn import (
    NeuralNetwork,
    fit_min_max_normalization,
    apply_min_max_normalization,
    run_forward_pass,
    compute_cross_entropy_cost,
)

DATASET = REPO_ROOT / "parkinsons.csv"
BEST_HIDDEN = [32]   # best architecture: 22-32-2 (highest accuracy in nn_10fold sweep)
BEST_LAMBDA = 0.00
LEARNING_RATE = 0.1
MAX_ITER = 5000
STEP_SIZE = 10
N_SEEDS = 5


def load_parkinsons(csv_path):
    import csv as _csv
    with open(csv_path, "r", newline="") as f:
        reader = _csv.reader(f)
        rows = [row for row in reader if row]
    header = rows[0]
    data_rows = rows[1:]
    data = np.array(data_rows, dtype=object)
    label_col = header.index("Diagnosis")
    feature_indices = [i for i in range(len(header)) if i != label_col]
    X = data[:, feature_indices].astype(float)
    y = data[:, label_col].astype(int)
    return X, y


def one_hot(y, n_classes=2):
    y_idx = np.asarray(y, dtype=int)
    return np.eye(n_classes)[y_idx]


def compute_J_after_fit(nn_model, X_train, y_onehot):
    """Compute training cost after fitting (uses stored normalization stats)."""
    X_norm = apply_min_max_normalization(X_train.astype(float), nn_model.normalization_stats)
    activations, _ = run_forward_pass(X_norm, nn_model.weight_matrices)
    return compute_cross_entropy_cost(
        y_onehot, activations[-1], nn_model.weight_matrices, nn_model.reg_lambda
    )


def main():
    run_start = time.time()
    X, y = load_parkinsons(DATASET)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    n_total = len(y)
    train_sizes = list(range(STEP_SIZE, n_total, STEP_SIZE))
    if train_sizes[-1] != n_total:
        train_sizes.append(n_total)

    layers = [X.shape[1]] + BEST_HIDDEN + [2]
    print(f"Architecture: {layers}, λ={BEST_LAMBDA}", flush=True)

    costs_per_size = []
    for s in train_sizes:
        costs_seeds = []
        for seed in range(N_SEEDS):
            X_sub = X[:s]
            y_sub = one_hot(y[:s])
            nn = NeuralNetwork(
                layers=layers,
                regularization=BEST_LAMBDA,
                learning_rate=LEARNING_RATE,
                max_iterations=MAX_ITER,
            )
            nn.fit(X_sub, y_sub, normalize=True)
            J = compute_J_after_fit(nn, X_sub, y_sub)
            costs_seeds.append(J)
        mean_J = float(np.mean(costs_seeds))
        costs_per_size.append(mean_J)
        print(f"  n={s:3d} | J={mean_J:.4f}", flush=True)

    out_dir = ROOT / "results" / "parkinsons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "nn_learning_curve.csv"
    with open(out_csv, "w", newline="") as f:
        import csv as _csv
        writer = _csv.writer(f)
        writer.writerow(["training_size", "mean_cost"])
        for s, J in zip(train_sizes, costs_per_size):
            writer.writerow([s, J])
    print(f"Saved: {out_csv}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, costs_per_size, marker="o", color="#2ca02c", linewidth=2)
    ax.set_xlabel("Number of training instances", fontsize=12)
    ax.set_ylabel("Cost J (cross-entropy)", fontsize=12)
    arch_str = "-".join(str(l) for l in layers)
    ax.set_title(
        f"NN Learning Curve on Parkinson's ({arch_str}, λ={BEST_LAMBDA})", fontsize=12
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    WRITEUP_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = WRITEUP_DIR / "nn_learning_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")
    print(f"Total elapsed: {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
