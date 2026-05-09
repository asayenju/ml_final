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
from sklearn.model_selection import StratifiedKFold

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.nn import NeuralNetwork

K_FOLDS = 10
RANDOM_STATE = 42
DATASET = REPO_ROOT / "parkinsons.csv"

# (hidden_sizes, lambda, label_for_table)
CONFIGS = [
    ([8],      0.00,  "22-8-2,  λ=0"),
    ([16],     0.00,  "22-16-2, λ=0"),
    ([16],     0.01,  "22-16-2, λ=0.01"),
    ([16],     0.10,  "22-16-2, λ=0.1"),
    ([16],     1.00,  "22-16-2, λ=1"),
    ([32],     0.00,  "22-32-2, λ=0"),
    ([32],     0.01,  "22-32-2, λ=0.01"),
    ([16, 8],  0.00,  "22-16-8-2, λ=0"),
    ([16, 8],  0.01,  "22-16-8-2, λ=0.01"),
]
MAX_ITER = 5000
LEARNING_RATE = 0.1


def load_parkinsons(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    header = rows[0]
    data_rows = rows[1:]
    data = np.array(data_rows, dtype=object)
    label_col = header.index("Diagnosis")
    feature_indices = [i for i in range(len(header)) if i != label_col]
    X = data[:, feature_indices].astype(float)
    y = data[:, label_col].astype(int)
    return X, y


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = np.mean(y_true == y_pred)
    labels = np.unique(y_true)
    f1_per_class = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1_per_class.append(2 * p * r / (p + r + 1e-12))
    return float(acc), float(np.mean(f1_per_class))


class NeuralNetworkWrapper:

    def __init__(self, hidden_sizes, reg_lambda, learning_rate, max_iterations):
        self.hidden_sizes = hidden_sizes
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self._nn = None
        self._classes = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        y_idx = np.array([np.where(self._classes == lbl)[0][0] for lbl in y])
        y_onehot = np.eye(n_classes)[y_idx]

        layers = [X.shape[1]] + self.hidden_sizes + [n_classes]
        self._nn = NeuralNetwork(
            layers=layers,
            regularization=self.reg_lambda,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
        )
        self._nn.fit(X, y_onehot, normalize=True)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        probs = self._nn.predict(X, normalize=True)
        return self._classes[np.argmax(probs, axis=1)]


def main():
    run_start = time.time()
    X, y = load_parkinsons(DATASET)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"Loaded: {len(y)} samples, {X.shape[1]} features", flush=True)

    result_rows = []
    for cfg_idx, (hidden, lam, label) in enumerate(CONFIGS, start=1):
        hp_start = time.time()
        print(f"\n[{cfg_idx}/{len(CONFIGS)}] {label}", flush=True)
        accs, f1s = [], []
        for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
            model = NeuralNetworkWrapper(
                hidden_sizes=hidden,
                reg_lambda=lam,
                learning_rate=LEARNING_RATE,
                max_iterations=MAX_ITER,
            )
            model.fit(X[tr], y[tr])
            y_pred = model.predict(X[te])
            acc, f1 = compute_metrics(y[te], y_pred)
            accs.append(acc)
            f1s.append(f1)
            print(f"  fold {fold_idx}/{K_FOLDS} | acc={acc:.4f} macro_f1={f1:.4f}", flush=True)

        row = {
            "architecture": label,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
        }
        result_rows.append(row)
        print(
            f"Done | acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} "
            f"f1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} "
            f"| {time.time()-hp_start:.1f}s",
            flush=True,
        )

    out_dir = ROOT / "results" / "parkinsons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "nn_hyperparams.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"\nSaved: {out_csv}")

    labels = [r["architecture"] for r in result_rows]
    acc_means = np.array([r["accuracy_mean"] for r in result_rows])
    acc_stds = np.array([r["accuracy_std"] for r in result_rows])
    f1_means = np.array([r["macro_f1_mean"] for r in result_rows])
    f1_stds = np.array([r["macro_f1_std"] for r in result_rows])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width / 2, acc_means, width, yerr=acc_stds, capsize=4,
                   color="#1f77b4", alpha=0.85, label="Accuracy")
    bars2 = ax.bar(x + width / 2, f1_means, width, yerr=f1_stds, capsize=4,
                   color="#d62728", alpha=0.85, label="Macro F1")
    best_idx = np.argmax(acc_means)
    bars1[best_idx].set_edgecolor("black")
    bars1[best_idx].set_linewidth(2)
    bars2[best_idx].set_edgecolor("black")
    bars2[best_idx].set_linewidth(2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Neural Network 10-fold CV on Parkinson's Disease Dataset", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    WRITEUP_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = WRITEUP_DIR / "nn_arch_accuracy.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")
    print(f"Total elapsed: {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
