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

from library import RandomForestClassifierScratch

DATASET = REPO_ROOT / "parkinsons.csv"
K_FOLDS = 10
RANDOM_STATE = 42
TREE_VALUES = [1, 5, 10, 20, 30, 40, 50]
MAX_DEPTH = 10


def load_parkinsons(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    header = rows[0]
    data_rows = rows[1:]
    data = np.array(data_rows, dtype=object)
    label_col = header.index("Diagnosis")
    feature_indices = [i for i in range(len(header)) if i != label_col]
    X_raw = data[:, feature_indices]
    y = data[:, label_col].astype(int).astype(str)
    return X_raw, y, feature_indices


def preprocess_numeric(X_train, X_test):
    """Convert to float, imputing missing with training mean."""
    X_tr = np.array(X_train, dtype=object).copy()
    X_te = np.array(X_test, dtype=object).copy()
    for j in range(X_train.shape[1]):
        train_vals = np.array([float(v) for v in X_train[:, j]], dtype=float)
        test_vals = np.array([float(v) for v in X_test[:, j]], dtype=float)
        X_tr[:, j] = train_vals
        X_te[:, j] = test_vals
    return X_tr, X_te


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
    X_raw, y, feature_indices = load_parkinsons(DATASET)
    numeric_cols = set(range(len(feature_indices)))
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(
        f"Loaded: {len(y)} samples, {X_raw.shape[1]} features | "
        f"numeric_cols={len(numeric_cols)} | tree_values={TREE_VALUES}",
        flush=True,
    )

    rows = []
    for hp_idx, trees in enumerate(TREE_VALUES, start=1):
        hp_start = time.time()
        print(f"\n[{hp_idx}/{len(TREE_VALUES)}] n_trees={trees}", flush=True)
        accs, f1s = [], []
        for fold_idx, (tr, te) in enumerate(skf.split(X_raw, y), start=1):
            X_train, X_test = X_raw[tr], X_raw[te]
            y_train, y_test = y[tr], y[te]
            X_tr_p, X_te_p = preprocess_numeric(X_train, X_test)
            model = RandomForestClassifierScratch(
                n_trees=trees,
                max_depth=MAX_DEPTH,
                min_size=3,
                min_gain=1e-4,
                numeric_cols=numeric_cols,
                random_state=RANDOM_STATE,
            ).fit(X_tr_p, y_train)
            y_pred = model.predict(X_te_p)
            acc, f1 = compute_metrics(y_test, y_pred)
            accs.append(acc)
            f1s.append(f1)
            print(f"  fold {fold_idx}/{K_FOLDS} | acc={acc:.4f} macro_f1={f1:.4f}", flush=True)

        row = {
            "n_trees": trees,
            "max_depth": MAX_DEPTH,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
        }
        rows.append(row)
        print(
            f"[{hp_idx}/{len(TREE_VALUES)}] n_trees={trees} done | "
            f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} "
            f"macro_f1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} "
            f"| {time.time()-hp_start:.1f}s",
            flush=True,
        )

    out_dir = ROOT / "results" / "parkinsons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "rf_hyperparams.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")

    tree_vals = np.array([r["n_trees"] for r in rows])
    acc_means = np.array([r["accuracy_mean"] for r in rows])
    acc_stds = np.array([r["accuracy_std"] for r in rows])
    f1_means = np.array([r["macro_f1_mean"] for r in rows])
    f1_stds = np.array([r["macro_f1_std"] for r in rows])

    best_idx = np.argmax(acc_means)
    best_trees = int(tree_vals[best_idx])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tree_vals, acc_means, marker="o", color="#1f77b4", linewidth=2, label="Accuracy")
    ax.fill_between(tree_vals, acc_means - acc_stds, acc_means + acc_stds, color="#1f77b4", alpha=0.15)
    ax.plot(tree_vals, f1_means, marker="s", color="#d62728", linewidth=2, label="Macro F1")
    ax.fill_between(tree_vals, f1_means - f1_stds, f1_means + f1_stds, color="#d62728", alpha=0.15)
    ax.axvline(best_trees, linestyle="--", color="gray", alpha=0.7, label=f"Best n_trees={best_trees}")
    ax.set_xlabel("Number of trees", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Random Forest 10-fold CV on Parkinson's Disease Dataset", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    WRITEUP_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = WRITEUP_DIR / "rf_ntrees_accuracy.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")
    print(f"Total elapsed: {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
