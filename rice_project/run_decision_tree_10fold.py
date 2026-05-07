import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library import DecisionTreeClassifierScratch

DATASET = REPO_ROOT / "rice.csv"
K_FOLDS = 10
RANDOM_STATE = 42
TREE_VALUES = [3, 4, 5, 6, 8, 10, 12, 15]


def load_dataset(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)

    header = rows[0]
    has_header = any(h.lower() == "label" for h in header) or any(h.lower().startswith("attr") for h in header)
    data_rows = rows[1:] if has_header else rows
    data = np.array(data_rows, dtype=object)

    if has_header:
        label_idx = next((i for i, h in enumerate(header) if h.lower() == "label"), len(header) - 1)
    else:
        label_idx = data.shape[1] - 1

    feature_indices = [i for i in range(data.shape[1]) if i != label_idx]
    X = data[:, feature_indices]
    y = data[:, label_idx]
    return X, y


def infer_numeric_cols(X):
    numeric_cols = []
    for j in range(X.shape[1]):
        ok = True
        for v in X[:, j]:
            s = str(v).strip()
            if s == "":
                continue
            try:
                float(s)
            except ValueError:
                ok = False
                break
        if ok:
            numeric_cols.append(j)
    return numeric_cols


def preprocess_for_decision_tree(X_train, X_test, numeric_cols, bins):
    X_train_out = np.array(X_train, dtype=object).copy()
    X_test_out = np.array(X_test, dtype=object).copy()

    for j in range(X_train.shape[1]):
        if j in numeric_cols:
            train_vals = np.array([float(v) if str(v).strip() != "" else np.nan for v in X_train[:, j]], dtype=float)
            mean = np.nanmean(train_vals) if np.any(~np.isnan(train_vals)) else 0.0
            train_vals = np.where(np.isnan(train_vals), mean, train_vals)

            quantiles = np.linspace(0, 1, bins + 1)[1:-1]
            edges = np.unique(np.quantile(train_vals, quantiles)) if bins > 1 else np.array([])

            X_train_out[:, j] = np.digitize(train_vals, edges, right=False).astype(str)

            test_vals = np.array([float(v) if str(v).strip() != "" else np.nan for v in X_test[:, j]], dtype=float)
            test_vals = np.where(np.isnan(test_vals), mean, test_vals)
            X_test_out[:, j] = np.digitize(test_vals, edges, right=False).astype(str)
        else:
            X_train_out[:, j] = np.array([str(v).strip() if str(v).strip() != "" else "MISSING" for v in X_train[:, j]], dtype=object)
            X_test_out[:, j] = np.array([str(v).strip() if str(v).strip() != "" else "MISSING" for v in X_test[:, j]], dtype=object)

    return X_train_out, X_test_out


def metrics_weighted(y_true, y_pred):
    labels, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    acc = np.mean(y_true == y_pred)
    prec_sum = 0.0
    rec_sum = 0.0
    f1_sum = 0.0
    for label, count in zip(labels, counts):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        w = count / total
        prec_sum += p * w
        rec_sum += r * w
        f1_sum += f1 * w
    return acc, prec_sum, rec_sum, f1_sum


def main():
    X, y = load_dataset(DATASET)
    numeric_cols = infer_numeric_cols(X)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for trees in TREE_VALUES:
        accs, precs, recs, f1s = [], [], [], []
        for tr, te in skf.split(X, y):
            X_train, X_test = X[tr], X[te]
            y_train, y_test = y[tr], y[te]
            X_train_p, X_test_p = preprocess_for_decision_tree(X_train, X_test, numeric_cols, trees)
            model = DecisionTreeClassifierScratch().fit(X_train_p, y_train)
            y_pred = model.predict(X_test_p)
            acc, p, r, f1 = metrics_weighted(y_test, y_pred)
            accs.append(acc)
            precs.append(p)
            recs.append(r)
            f1s.append(f1)

        rows.append({
            "trees": trees,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "precision_mean": float(np.mean(precs)),
            "precision_std": float(np.std(precs)),
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        })
        print(f"trees={trees}: acc={np.mean(accs):.4f}, prec={np.mean(precs):.4f}, rec={np.mean(recs):.4f}, f1={np.mean(f1s):.4f}")

    out_dir = ROOT / "results" / "rice"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "decision_tree_hyperparams_10fold.csv"
    headers = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[h] for h in headers])

    x = [r["trees"] for r in rows]
    for metric, color in [("accuracy", "#1f77b4"), ("precision", "#ff7f0e"), ("recall", "#2ca02c"), ("f1", "#d62728")]:
        yv = [r[f"{metric}_mean"] for r in rows]
        ys = [r[f"{metric}_std"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, yv, marker="o", color=color)
        ax.fill_between(x, np.array(yv)-np.array(ys), np.array(yv)+np.array(ys), color=color, alpha=0.15)
        ax.set_xlabel("Number of trees")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Decision Tree ({K_FOLDS}-fold) on Rice: {metric.upper()} vs trees")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"decision_tree_{metric}_vs_trees_10fold.png", dpi=150)
        plt.close(fig)

    print(f"Saved: {out_csv}")
    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
