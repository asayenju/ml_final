

import csv
import os
from pathlib import Path
import sys

ROOT      = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.knn           import KNNClassifier
from library.random_forest import RandomForestClassifierScratch
from library.nn            import NeuralNetwork

DATASET      = REPO_ROOT / "ObesityDataSet_raw_and_data_sinthetic.csv"
K_FOLDS      = 10
RANDOM_STATE = 42

KNN_K_VALUES = list(range(1, 52, 2))

RF_CONFIGS = [
    (5,  5),  (10, 5),  (15, 8),  (20, 8),
    (20, 10), (30, 10), (50, 12),
]

C_BLUE   = "#2563EB"
C_GREEN  = "#16A34A"
C_PURPLE = "#7C3AED"
C_RED    = "#DC2626"
C_GREY   = "#6B7280"
C_BG     = "#F8FAFC"
C_GRID   = "#E2E8F0"


def load_data(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)

    header    = rows[0]
    data_rows = rows[1:]
    data      = np.array(data_rows, dtype=object)

    label_idx       = len(header) - 1
    feature_indices = list(range(label_idx))

    X_raw = data[:, feature_indices]
    y_raw = data[:, label_idx]

    classes = sorted(set(y_raw))
    cls2int = {c: i for i, c in enumerate(classes)}
    y       = np.array([cls2int[v] for v in y_raw])

    cat_cols = [1, 4, 5, 8, 9, 11, 14, 15]
    num_idxs = set(i for i in range(X_raw.shape[1]) if i not in cat_cols)

    print(f"Loaded : {X_raw.shape[0]} samples | {X_raw.shape[1]} raw features | "
          f"{len(classes)} classes")
    print(f"Classes: {classes}\n")
    return X_raw, y, classes, num_idxs, cat_cols


def build_global_categories(X_raw, cat_cols):
    return {j: sorted(set(X_raw[:, j])) for j in cat_cols}


def ohe_encode(X_raw, cat_cols, fit_categories):
    d        = X_raw.shape[1]
    num_cols = [j for j in range(d) if j not in cat_cols]

    parts = []
    for j in num_cols:
        col = np.array([float(v) for v in X_raw[:, j]], dtype=float).reshape(-1, 1)
        parts.append(col)
    for j in cat_cols:
        for cat in fit_categories[j]:
            col = (X_raw[:, j] == cat).astype(float).reshape(-1, 1)
            parts.append(col)

    return np.hstack(parts)


def stratified_kfold(y, k=10, seed=42):
    rng     = np.random.default_rng(seed)
    classes = np.unique(y)
    ci      = {c: rng.permutation(np.where(y == c)[0]) for c in classes}
    cf      = {c: np.array_split(ci[c], k)             for c in classes}
    folds   = []
    for i in range(k):
        test  = np.concatenate([cf[c][i] for c in classes])
        train = np.concatenate([
            np.concatenate([cf[c][j] for j in range(k) if j != i])
            for c in classes
        ])
        folds.append((train, test))
    return folds


def accuracy(yt, yp):
    return float(np.mean(yt == yp))

def f1_macro(yt, yp):
    f1s = []
    for c in np.unique(np.concatenate([yt, yp])):
        tp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        p  = tp / (tp + fp) if tp + fp else 0.0
        r  = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def run_knn(X_raw, y, folds, cat_cols, global_cats):
    results = []
    for k in KNN_K_VALUES:
        accs, f1s = [], []
        for tr, te in folds:
            X_tr = ohe_encode(X_raw[tr], cat_cols, global_cats)
            X_te = ohe_encode(X_raw[te], cat_cols, global_cats)
            m    = KNNClassifier(k=k).fit(X_tr, y[tr])
            p    = m.predict(X_te).astype(int)
            accs.append(accuracy(y[te], p))
            f1s.append(f1_macro(y[te], p))
        results.append(dict(k=k,
                            acc=np.mean(accs), acc_std=np.std(accs),
                            f1=np.mean(f1s),   f1_std=np.std(f1s)))
        print(f"  KNN k={k:2d}  acc={results[-1]['acc']:.4f}  f1={results[-1]['f1']:.4f}")
    return results


def run_rf(X_raw, y, folds, num_idxs):
    results = []
    for n_trees, max_depth in RF_CONFIGS:
        accs, f1s = [], []
        for tr, te in folds:
            m = RandomForestClassifierScratch(
                n_trees=n_trees, max_depth=max_depth,
                min_size=3, min_gain=1e-4,
                numeric_cols=num_idxs, random_state=RANDOM_STATE,
            ).fit(X_raw[tr], y[tr])
            p = m.predict(X_raw[te]).astype(int)
            accs.append(accuracy(y[te], p))
            f1s.append(f1_macro(y[te], p))
        label = f"T={n_trees}\nD={max_depth}"
        results.append(dict(label=label, n_trees=n_trees, max_depth=max_depth,
                            acc=np.mean(accs), acc_std=np.std(accs),
                            f1=np.mean(f1s),   f1_std=np.std(f1s)))
        print(f"  RF  T={n_trees:2d} D={max_depth:2d}  acc={results[-1]['acc']:.4f}  f1={results[-1]['f1']:.4f}")
    return results


def run_nn(X_raw, y, folds, cat_cols, n_classes, global_cats):
    n_in = ohe_encode(X_raw[:1], cat_cols, global_cats).shape[1]

    configs = [
        dict(arch=(n_in, 16, n_classes),         reg=0.0,  iters=2000),
        dict(arch=(n_in, 32, n_classes),         reg=0.0,  iters=2000),
        dict(arch=(n_in, 64, n_classes),         reg=0.0,  iters=2000),
        dict(arch=(n_in, 32, n_classes),         reg=0.01, iters=2000),
        dict(arch=(n_in, 32, n_classes),         reg=0.1,  iters=2000),
        dict(arch=(n_in, 32, 16, n_classes),     reg=0.01, iters=2000),
        dict(arch=(n_in, 64, 32, n_classes),     reg=0.01, iters=2000),
    ]
    results = []
    for cfg in configs:
        accs, f1s = [], []
        for tr, te in folds:
            X_tr = ohe_encode(X_raw[tr], cat_cols, global_cats)
            X_te = ohe_encode(X_raw[te], cat_cols, global_cats)
            Y    = np.zeros((len(tr), n_classes))
            Y[np.arange(len(tr)), y[tr]] = 1
            m = NeuralNetwork(
                layers=cfg["arch"], regularization=cfg["reg"],
                learning_rate=0.05, max_iterations=cfg["iters"],
            ).fit(X_tr, Y)
            p = np.argmax(m.predict(X_te), axis=1)
            accs.append(accuracy(y[te], p))
            f1s.append(f1_macro(y[te], p))
        arch_s = "-".join(str(s) for s in cfg["arch"])
        label  = f"{arch_s}\nreg={cfg['reg']}"
        results.append(dict(label=label,
                            acc=np.mean(accs), acc_std=np.std(accs),
                            f1=np.mean(f1s),   f1_std=np.std(f1s)))
        print(f"  NN  [{arch_s}] reg={cfg['reg']}  acc={results[-1]['acc']:.4f}  f1={results[-1]['f1']:.4f}")
    return results


def save_csv(rows, path, key_order):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(key_order)
        for row in rows:
            writer.writerow([row[k] for k in key_order])
    print(f"  → CSV : {path}")


def _base_style(fig, ax):
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(labelsize=9)


def plot_knn(results, out_dir):
    ks   = [r["k"]       for r in results]
    accs = [r["acc"]     for r in results]
    f1s  = [r["f1"]      for r in results]
    astd = [r["acc_std"] for r in results]
    fstd = [r["f1_std"]  for r in results]

    accs_no1 = [a if k > 1 else -1 for k, a in zip(ks, accs)]
    bi        = int(np.argmax(accs_no1))

    fig, ax = plt.subplots(figsize=(12, 5))
    _base_style(fig, ax)
    ax.grid(color=C_GRID, linewidth=1, zorder=0)

    ax.fill_between(ks, np.array(accs) - np.array(astd),
                        np.array(accs) + np.array(astd),
                    color=C_BLUE, alpha=0.12, zorder=2)
    ax.fill_between(ks, np.array(f1s) - np.array(fstd),
                        np.array(f1s) + np.array(fstd),
                    color=C_GREEN, alpha=0.12, zorder=2)
    ax.plot(ks, accs, marker="o", markersize=4, color=C_BLUE,
            lw=2, label="Accuracy", zorder=3)
    ax.plot(ks, f1s,  marker="s", markersize=4, color=C_GREEN,
            lw=2, label="Macro F1", zorder=3, linestyle="--")

    ax.axvline(1, color=C_RED, linestyle=":", alpha=0.6, lw=1.4)
    ax.annotate("k=1: synthetic\ndata artefact",
                xy=(1, accs[0]), xytext=(4, accs[0] - 0.06),
                fontsize=8, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.1))

    ax.axvline(ks[bi], color=C_BLUE, linestyle=":", alpha=0.5, lw=1.4)
    ax.annotate(f"Best k={ks[bi]}\nacc={accs[bi]:.3f}",
                xy=(ks[bi], accs[bi]),
                xytext=(ks[bi] + 2, accs[bi] - 0.06),
                fontsize=8, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.1))

    ax.set_xlabel("Number of Neighbours  k  (odd values only)", fontsize=12)
    ax.set_ylabel("Score  (10-fold stratified CV)", fontsize=12)
    ax.set_title("K-Nearest Neighbours – Performance vs k\nObesity Dataset",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(ks[::2])
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = out_dir / "knn_performance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Plot: {path}")


def plot_rf(results, out_dir):
    labels = [r["label"]   for r in results]
    accs   = [r["acc"]     for r in results]
    f1s    = [r["f1"]      for r in results]
    astd   = [r["acc_std"] for r in results]
    fstd   = [r["f1_std"]  for r in results]
    x      = np.arange(len(labels))
    bi     = int(np.argmax(accs))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    _base_style(fig, ax)
    ax.grid(color=C_GRID, linewidth=1, axis="y", zorder=0)

    ba = ax.bar(x - w/2, accs, w, color=C_BLUE,  alpha=0.85, label="Accuracy",
                yerr=astd, capsize=3, error_kw=dict(ecolor=C_GREY, lw=1.2), zorder=3)
    bf = ax.bar(x + w/2, f1s,  w, color=C_GREEN, alpha=0.85, label="Macro F1",
                yerr=fstd, capsize=3, error_kw=dict(ecolor=C_GREY, lw=1.2), zorder=3)

    for b in (ba[bi], bf[bi]):
        b.set_edgecolor(C_PURPLE)
        b.set_linewidth(2.5)
    ax.text(bi - w/2, accs[bi] + astd[bi] + 0.008,
            f"★ best\n{accs[bi]:.3f}", ha="center", fontsize=7.5,
            color=C_PURPLE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Score  (10-fold stratified CV)", fontsize=12)
    ax.set_title("Random Forest – Performance vs Hyperparameters\nObesity Dataset",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = out_dir / "rf_performance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Plot: {path}")


def plot_nn(results, out_dir):
    labels = [r["label"]   for r in results]
    accs   = [r["acc"]     for r in results]
    f1s    = [r["f1"]      for r in results]
    astd   = [r["acc_std"] for r in results]
    fstd   = [r["f1_std"]  for r in results]
    x      = np.arange(len(labels))
    bi     = int(np.argmax(accs))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    _base_style(fig, ax)
    ax.grid(color=C_GRID, linewidth=1, axis="y", zorder=0)

    ba = ax.bar(x - w/2, accs, w, color=C_PURPLE, alpha=0.85, label="Accuracy",
                yerr=astd, capsize=3, error_kw=dict(ecolor=C_GREY, lw=1.2), zorder=3)
    bf = ax.bar(x + w/2, f1s,  w, color=C_GREEN,  alpha=0.85, label="Macro F1",
                yerr=fstd, capsize=3, error_kw=dict(ecolor=C_GREY, lw=1.2), zorder=3)

    for b in (ba[bi], bf[bi]):
        b.set_edgecolor(C_RED)
        b.set_linewidth(2.5)
    ax.text(bi - w/2, accs[bi] + astd[bi] + 0.008,
            f"★ best\n{accs[bi]:.3f}", ha="center", fontsize=7.5,
            color=C_RED, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Score  (10-fold stratified CV)", fontsize=12)
    ax.set_title("Neural Network – Performance vs Architecture & Regularisation\nObesity Dataset",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = out_dir / "nn_performance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Plot: {path}")


def print_summary(knn_res, rf_res, nn_res):
    def best(res, skip_k1=False):
        scores = [r["acc"] for r in res]
        if skip_k1:
            scores = [-1 if r["k"] == 1 else s for r, s in zip(res, scores)]
        bi = int(np.argmax(scores))
        return res[bi]

    kb = best(knn_res, skip_k1=True)
    rb = best(rf_res)
    nb = best(nn_res)

    print("\n" + "=" * 68)
    print(f"{'Algorithm':<28} {'Best Setting':<16} {'Accuracy':>9}  {'Macro-F1':>9}")
    print("-" * 68)
    print(f"{'K-NN (k=1 excluded*)':<28} {'k='+str(kb['k']):<16} {kb['acc']:>9.4f}  {kb['f1']:>9.4f}")
    print(f"{'Random Forest':<28} {'T='+str(rb['n_trees'])+' D='+str(rb['max_depth']):<16} {rb['acc']:>9.4f}  {rb['f1']:>9.4f}")
    print(f"{'Neural Network':<28} {'':<16} {nb['acc']:>9.4f}  {nb['f1']:>9.4f}")
    print("=" * 68)
    print("* k=1 wins due to 74.8% SMOTE-synthetic data; excluded from best selection.")


def main():
    X_raw, y, classes, num_idxs, cat_cols = load_data(DATASET)
    n_classes   = len(classes)
    global_cats = build_global_categories(X_raw, cat_cols)
    folds       = stratified_kfold(y, k=K_FOLDS, seed=RANDOM_STATE)
    out_dir     = ROOT / "results" / "obesity"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"▶ K-NN ({K_FOLDS}-fold, k = 1 to 51 odd) …")
    knn_res = run_knn(X_raw, y, folds, cat_cols, global_cats)
    plot_knn(knn_res, out_dir)
    save_csv(knn_res, out_dir / "knn_10fold.csv",
             ["k", "acc", "acc_std", "f1", "f1_std"])

    print(f"\n▶ Random Forest ({K_FOLDS}-fold, {len(RF_CONFIGS)} configs) …")
    rf_res = run_rf(X_raw, y, folds, num_idxs)
    plot_rf(rf_res, out_dir)
    save_csv(rf_res, out_dir / "rf_10fold.csv",
             ["label", "n_trees", "max_depth", "acc", "acc_std", "f1", "f1_std"])

    print(f"\n▶ Neural Network ({K_FOLDS}-fold, 7 architectures) …")
    nn_res = run_nn(X_raw, y, folds, cat_cols, n_classes, global_cats)
    plot_nn(nn_res, out_dir)
    save_csv(nn_res, out_dir / "nn_10fold.csv",
             ["label", "acc", "acc_std", "f1", "f1_std"])

    print_summary(knn_res, rf_res, nn_res)
    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()