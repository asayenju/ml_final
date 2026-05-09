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

from library import KNNClassifier

DATASET = REPO_ROOT / "parkinsons.csv"
K_FOLDS = 10
RANDOM_STATE = 42
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 19, 25, 31, 37, 43, 49]


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
    y = data[:, label_col].astype(int).astype(str)
    return X, y


def normalize_zscore(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < 1e-12] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


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


def main():
    run_start = time.time()
    X, y = load_parkinsons(DATASET)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(
        f"Loaded: {len(y)} samples, {X.shape[1]} features | "
        f"classes={np.unique(y, return_counts=True)}",
        flush=True,
    )

    rows = []
    for k in K_VALUES:
        accs, f1s = [], []
        for tr, te in skf.split(X, y):
            X_train, X_test = X[tr], X[te]
            y_train, y_test = y[tr], y[te]
            X_tr_n, X_te_n = normalize_zscore(X_train, X_test)
            model = KNNClassifier(k=k).fit(X_tr_n, y_train)
            y_pred = model.predict(X_te_n)
            acc, f1 = compute_metrics(y_test, y_pred)
            accs.append(acc)
            f1s.append(f1)

        row = {
            "k": k,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
        }
        rows.append(row)
        print(
            f"k={k:2d} | acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} "
            f"macro_f1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}",
            flush=True,
        )

    out_dir = ROOT / "results" / "parkinsons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "knn_hyperparams.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")

    k_vals = np.array([r["k"] for r in rows])
    acc_means = np.array([r["accuracy_mean"] for r in rows])
    acc_stds = np.array([r["accuracy_std"] for r in rows])
    f1_means = np.array([r["macro_f1_mean"] for r in rows])
    f1_stds = np.array([r["macro_f1_std"] for r in rows])

    best_idx = np.argmax(acc_means)
    best_k = int(k_vals[best_idx])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_vals, acc_means, marker="o", color="#1f77b4", linewidth=2, label="Accuracy")
    ax.fill_between(k_vals, acc_means - acc_stds, acc_means + acc_stds, color="#1f77b4", alpha=0.15)
    ax.plot(k_vals, f1_means, marker="s", color="#d62728", linewidth=2, label="Macro F1")
    ax.fill_between(k_vals, f1_means - f1_stds, f1_means + f1_stds, color="#d62728", alpha=0.15)
    ax.axvline(best_k, linestyle="--", color="gray", alpha=0.7, label=f"Best k={best_k}")
    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.set_title("KNN 10-fold CV on Parkinson's Disease Dataset", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    WRITEUP_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = WRITEUP_DIR / "knn_k_accuracy.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")
    print(f"Total elapsed: {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
