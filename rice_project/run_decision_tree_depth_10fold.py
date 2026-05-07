import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library import DecisionTreeClassifierScratch

K_FOLDS = 10
RANDOM_STATE = 42
DEPTH_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]


def weighted_metrics(y_true, y_pred):
    labels, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    acc = np.mean(y_true == y_pred)
    p_sum = 0.0
    r_sum = 0.0
    f1_sum = 0.0
    for label, count in zip(labels, counts):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        w = count / total
        p_sum += p * w
        r_sum += r * w
        f1_sum += f1 * w
    return acc, p_sum, r_sum, f1_sum


def main():
    data_rows = []
    with open(ROOT / 'rice.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                data_rows.append(row)

    header = data_rows[0]
    label_idx = next((i for i, h in enumerate(header) if h.lower() == 'label'), len(header) - 1)
    raw = np.array(data_rows[1:], dtype=object)
    feature_indices = [i for i in range(raw.shape[1]) if i != label_idx]
    X = raw[:, feature_indices].astype(float)
    y = raw[:, label_idx]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for depth in DEPTH_VALUES:
        accs, precs, recs, f1s = [], [], [], []
        for tr_idx, te_idx in skf.split(X, y):
            X_train, X_test = X[tr_idx], X[te_idx]
            y_train, y_test = y[tr_idx], y[te_idx]

            # Convert numeric features to bins per fold, then train ID3 tree on categories.
            X_train_b = np.empty_like(X_train, dtype=object)
            X_test_b = np.empty_like(X_test, dtype=object)
            for j in range(X_train.shape[1]):
                quantiles = np.linspace(0, 1, depth + 1)[1:-1]
                edges = np.unique(np.quantile(X_train[:, j], quantiles)) if depth > 1 else np.array([])
                X_train_b[:, j] = np.digitize(X_train[:, j], edges, right=False).astype(str)
                X_test_b[:, j] = np.digitize(X_test[:, j], edges, right=False).astype(str)

            model = DecisionTreeClassifierScratch().fit(X_train_b, y_train)
            y_pred = model.predict(X_test_b)

            acc, p, r, f1 = weighted_metrics(y_test, y_pred)
            accs.append(acc)
            precs.append(p)
            recs.append(r)
            f1s.append(f1)

        rows.append({
            'max_depth': depth,
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'precision_mean': float(np.mean(precs)),
            'precision_std': float(np.std(precs)),
            'recall_mean': float(np.mean(recs)),
            'recall_std': float(np.std(recs)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
        })
        print(f"depth={depth}: acc={np.mean(accs):.4f}, prec={np.mean(precs):.4f}, rec={np.mean(recs):.4f}, f1={np.mean(f1s):.4f}")

    out_dir = ROOT / 'results' / 'rice'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'decision_tree_depth_hyperparams_10fold.csv'

    headers = list(rows[0].keys())
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[h] for h in headers])

    x = [r['max_depth'] for r in rows]
    for metric, color in [('accuracy', '#1f77b4'), ('precision', '#ff7f0e'), ('recall', '#2ca02c'), ('f1', '#d62728')]:
        yv = [r[f'{metric}_mean'] for r in rows]
        ys = [r[f'{metric}_std'] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, yv, marker='o', linewidth=2, color=color)
        ax.fill_between(x, np.array(yv) - np.array(ys), np.array(yv) + np.array(ys), color=color, alpha=0.15)
        ax.set_xlabel('Max Depth')
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Decision Tree (10-fold) on Rice: {metric.upper()} vs max_depth')
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f'decision_tree_{metric}_vs_depth_10fold.png', dpi=150)
        plt.close(fig)

    print(f'Saved: {out_csv}')
    print(f'Saved plots in: {out_dir}')


if __name__ == '__main__':
    main()
