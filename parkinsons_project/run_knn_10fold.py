import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library import KNNClassifier

DATASET = REPO_ROOT / "parkinsons.csv"
LABEL_COL = "Diagnosis"
K_FOLDS = 10
RANDOM_STATE = 42
K_VALUES = [1, 3, 5, 7, 9, 11, 15]


def load_dataset(csv_path, label_col):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)

    header = rows[0]
    data = np.array(rows[1:], dtype=object)
    label_idx = header.index(label_col)

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


def preprocess_for_knn(X_train, X_test, numeric_cols):
    train_num = []
    test_num = []

    for j in numeric_cols:
        train_vals = np.array([float(v) if str(v).strip() != "" else np.nan for v in X_train[:, j]], dtype=float)
        mean = np.nanmean(train_vals) if np.any(~np.isnan(train_vals)) else 0.0
        train_vals = np.where(np.isnan(train_vals), mean, train_vals)

        std = np.std(train_vals)
        std = std if std > 1e-12 else 1.0

        test_vals = np.array([float(v) if str(v).strip() != "" else np.nan for v in X_test[:, j]], dtype=float)
        test_vals = np.where(np.isnan(test_vals), mean, test_vals)

        train_num.append(((train_vals - mean) / std).reshape(-1, 1))
        test_num.append(((test_vals - mean) / std).reshape(-1, 1))

    if not train_num:
        return np.zeros((len(X_train), 1), dtype=float), np.zeros((len(X_test), 1), dtype=float)

    return np.hstack(train_num), np.hstack(test_num)


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
    X, y = load_dataset(DATASET, LABEL_COL)
    numeric_cols = infer_numeric_cols(X)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for k in K_VALUES:
        accs, precs, recs, f1s = [], [], [], []
        for tr, te in skf.split(X, y):
            X_train, X_test = X[tr], X[te]
            y_train, y_test = y[tr], y[te]
            X_train_p, X_test_p = preprocess_for_knn(X_train, X_test, numeric_cols)
            model = KNNClassifier(k=k).fit(X_train_p, y_train)
            y_pred = model.predict(X_test_p)
            acc, p, r, f1 = metrics_weighted(y_test, y_pred)
            accs.append(acc)
            precs.append(p)
            recs.append(r)
            f1s.append(f1)

        rows.append({
            "k": k,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "precision_mean": float(np.mean(precs)),
            "precision_std": float(np.std(precs)),
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        })
        print(f"k={k}: acc={np.mean(accs):.4f}, prec={np.mean(precs):.4f}, rec={np.mean(recs):.4f}, f1={np.mean(f1s):.4f}")

    out_dir = ROOT / "results" / "parkinsons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "knn_hyperparams_10fold.csv"
    headers = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[h] for h in headers])

    x = [r["k"] for r in rows]
    for metric, color in [("accuracy", "#1f77b4"), ("precision", "#ff7f0e"), ("recall", "#2ca02c"), ("f1", "#d62728")]:
        yv = [r[f"{metric}_mean"] for r in rows]
        ys = [r[f"{metric}_std"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, yv, marker="o", color=color)
        ax.fill_between(x, np.array(yv) - np.array(ys), np.array(yv) + np.array(ys), color=color, alpha=0.15)
        ax.set_xlabel("k (number of neighbors)")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.set_title(f"KNN ({K_FOLDS}-fold) on Parkinsons: {metric.upper()} vs k")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"knn_{metric}_vs_k_10fold.png", dpi=150)
        plt.close(fig)

    print(f"Saved: {out_csv}")
    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
