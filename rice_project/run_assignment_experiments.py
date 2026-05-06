import csv
import os
import argparse
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

from library import KNNClassifier, DecisionTreeClassifierScratch, RandomForestClassifierScratch


DATASET_CANDIDATES = [
    REPO_ROOT / "rice.csv",
    REPO_ROOT / "credit_approval.csv",
    REPO_ROOT / "parkinsons.csv",
    REPO_ROOT / "raisin.csv",
]

K_FOLDS = 10
RANDOM_STATE = 42

KNN_K_VALUES = [1, 3, 5, 7, 9, 11]
DT_BIN_VALUES = [3, 4, 5, 6, 8, 10]
RF_NTREE_VALUES = [1, 5, 10, 20, 30, 50]


def weighted_f1_score(y_true, y_pred):
    labels, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    score_sum = 0.0
    for label, count in zip(labels, counts):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        score_sum += f1 * (count / total)
    return score_sum


def load_dataset(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)

    if not rows:
        return np.empty((0, 0), dtype=object), np.empty((0,), dtype=object), []

    header = rows[0]
    has_header = any(h.lower() == "label" for h in header) or any(h.lower().startswith("attr") for h in header)
    data_rows = rows[1:] if has_header else rows
    if not data_rows:
        return np.empty((0, 0), dtype=object), np.empty((0,), dtype=object), header if has_header else []

    data = np.array(data_rows, dtype=object)

    if has_header:
        label_idx = next((i for i, h in enumerate(header) if h.lower() == "label"), len(header) - 1)
    else:
        label_idx = data.shape[1] - 1

    feature_indices = [i for i in range(data.shape[1]) if i != label_idx]
    X = data[:, feature_indices]
    y = data[:, label_idx]
    feature_names = [header[i] for i in feature_indices] if has_header else [f"f{i}" for i in range(len(feature_indices))]
    return X, y, feature_names


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
    n_features = X_train.shape[1]
    categorical_cols = [j for j in range(n_features) if j not in numeric_cols]

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

    train_cat = []
    test_cat = []
    for j in categorical_cols:
        train_col = np.array([str(v).strip() if str(v).strip() != "" else "MISSING" for v in X_train[:, j]], dtype=object)
        test_col = np.array([str(v).strip() if str(v).strip() != "" else "MISSING" for v in X_test[:, j]], dtype=object)
        uniq = np.unique(train_col)
        for u in uniq:
            train_cat.append((train_col == u).astype(float).reshape(-1, 1))
            test_cat.append((test_col == u).astype(float).reshape(-1, 1))

    blocks_train = train_num + train_cat
    blocks_test = test_num + test_cat
    if not blocks_train:
        return np.zeros((len(X_train), 1), dtype=float), np.zeros((len(X_test), 1), dtype=float)

    return np.hstack(blocks_train), np.hstack(blocks_test)


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

            train_bucket = np.digitize(train_vals, edges, right=False).astype(str)
            X_train_out[:, j] = train_bucket

            test_vals = np.array([float(v) if str(v).strip() != "" else np.nan for v in X_test[:, j]], dtype=float)
            test_vals = np.where(np.isnan(test_vals), mean, test_vals)
            test_bucket = np.digitize(test_vals, edges, right=False).astype(str)
            X_test_out[:, j] = test_bucket
        else:
            X_train_out[:, j] = np.array([str(v).strip() if str(v).strip() != "" else "MISSING" for v in X_train[:, j]], dtype=object)
            X_test_out[:, j] = np.array([str(v).strip() if str(v).strip() != "" else "MISSING" for v in X_test[:, j]], dtype=object)

    return X_train_out, X_test_out


def preprocess_for_random_forest(X_train, X_test, numeric_cols):
    X_train_out = np.array(X_train, dtype=object).copy()
    X_test_out = np.array(X_test, dtype=object).copy()

    for j in range(X_train.shape[1]):
        if j in numeric_cols:
            train_vals = np.array([float(v) if str(v).strip() != "" else np.nan for v in X_train[:, j]], dtype=float)
            mean = np.nanmean(train_vals) if np.any(~np.isnan(train_vals)) else 0.0
            train_vals = np.where(np.isnan(train_vals), mean, train_vals)
            X_train_out[:, j] = train_vals

            test_vals = np.array([float(v) if str(v).strip() != "" else np.nan for v in X_test[:, j]], dtype=float)
            test_vals = np.where(np.isnan(test_vals), mean, test_vals)
            X_test_out[:, j] = test_vals
        else:
            X_train_out[:, j] = np.array([str(v).strip() if str(v).strip() != "" else "MISSING" for v in X_train[:, j]], dtype=object)
            X_test_out[:, j] = np.array([str(v).strip() if str(v).strip() != "" else "MISSING" for v in X_test[:, j]], dtype=object)

    return X_train_out, X_test_out


def evaluate_hyperparams(X, y, numeric_cols, algorithm_name, param_values):
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for value in param_values:
        fold_acc = []
        fold_f1 = []

        for train_idx, test_idx in skf.split(X, y):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if algorithm_name == "knn":
                X_train, X_test = preprocess_for_knn(X_train_raw, X_test_raw, numeric_cols)
                model = KNNClassifier(k=value)
            elif algorithm_name == "decision_tree":
                X_train, X_test = preprocess_for_decision_tree(X_train_raw, X_test_raw, numeric_cols, bins=value)
                model = DecisionTreeClassifierScratch()
            elif algorithm_name == "random_forest":
                X_train, X_test = preprocess_for_random_forest(X_train_raw, X_test_raw, numeric_cols)
                model = RandomForestClassifierScratch(
                    n_trees=value,
                    max_depth=10,
                    min_size=3,
                    min_gain=1e-4,
                    numeric_cols=set(numeric_cols),
                    random_state=RANDOM_STATE,
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_acc.append(np.mean(y_pred == y_test))
            fold_f1.append(weighted_f1_score(y_test, y_pred))

        rows.append(
            {
                "value": value,
                "accuracy_mean": float(np.mean(fold_acc)),
                "accuracy_std": float(np.std(fold_acc)),
                "f1_mean": float(np.mean(fold_f1)),
                "f1_std": float(np.std(fold_f1)),
            }
        )

    return rows


def save_rows_csv(rows, headers, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[h] for h in headers])


def plot_hyperparam_curve(rows, x_label, title, out_path):
    x = [r["value"] for r in rows]
    acc = [r["accuracy_mean"] for r in rows]
    f1 = [r["f1_mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, acc, marker="o", label="Accuracy")
    ax.plot(x, f1, marker="s", label="F1-score")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def best_row(rows, key):
    best = rows[0]
    for r in rows[1:]:
        if r[key] > best[key]:
            best = r
    return best


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset stem names to run (e.g., rice,parkinsons). Default: all found.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = [p for p in DATASET_CANDIDATES if p.exists()]
    if args.datasets.strip():
        wanted = {x.strip() for x in args.datasets.split(",") if x.strip()}
        datasets = [p for p in datasets if p.stem in wanted]
    if not datasets:
        print("No dataset files found from DATASET_CANDIDATES.")
        return

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    summary_rows = []

    for ds_path in datasets:
        dataset_name = ds_path.stem
        print(f"\nRunning experiments for dataset: {dataset_name}")
        X, y, feature_names = load_dataset(ds_path)
        if len(y) == 0:
            print(f"  Skipping {dataset_name}: empty dataset")
            continue

        numeric_cols = infer_numeric_cols(X)
        print(f"  Samples={len(y)} Features={X.shape[1]} Numeric features={len(numeric_cols)}")

        knn_rows = evaluate_hyperparams(X, y, numeric_cols, "knn", KNN_K_VALUES)
        dt_rows = evaluate_hyperparams(X, y, numeric_cols, "decision_tree", DT_BIN_VALUES)
        rf_rows = evaluate_hyperparams(X, y, numeric_cols, "random_forest", RF_NTREE_VALUES)

        ds_out = out_dir / dataset_name
        ds_out.mkdir(exist_ok=True)

        save_rows_csv(knn_rows, ["value", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"], ds_out / "knn_hyperparams.csv")
        save_rows_csv(dt_rows, ["value", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"], ds_out / "decision_tree_hyperparams.csv")
        save_rows_csv(rf_rows, ["value", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"], ds_out / "random_forest_hyperparams.csv")

        plot_hyperparam_curve(knn_rows, "k", f"KNN on {dataset_name}", ds_out / "knn_curve.png")
        plot_hyperparam_curve(dt_rows, "num_bins", f"Decision Tree on {dataset_name}", ds_out / "decision_tree_curve.png")
        plot_hyperparam_curve(rf_rows, "n_trees", f"Random Forest on {dataset_name}", ds_out / "random_forest_curve.png")

        knn_best_acc = best_row(knn_rows, "accuracy_mean")
        knn_best_f1 = best_row(knn_rows, "f1_mean")
        dt_best_acc = best_row(dt_rows, "accuracy_mean")
        dt_best_f1 = best_row(dt_rows, "f1_mean")
        rf_best_acc = best_row(rf_rows, "accuracy_mean")
        rf_best_f1 = best_row(rf_rows, "f1_mean")

        summary_rows.extend(
            [
                {
                    "dataset": dataset_name,
                    "algorithm": "KNN",
                    "best_accuracy": knn_best_acc["accuracy_mean"],
                    "best_accuracy_param": knn_best_acc["value"],
                    "best_f1": knn_best_f1["f1_mean"],
                    "best_f1_param": knn_best_f1["value"],
                },
                {
                    "dataset": dataset_name,
                    "algorithm": "DecisionTree",
                    "best_accuracy": dt_best_acc["accuracy_mean"],
                    "best_accuracy_param": dt_best_acc["value"],
                    "best_f1": dt_best_f1["f1_mean"],
                    "best_f1_param": dt_best_f1["value"],
                },
                {
                    "dataset": dataset_name,
                    "algorithm": "RandomForest",
                    "best_accuracy": rf_best_acc["accuracy_mean"],
                    "best_accuracy_param": rf_best_acc["value"],
                    "best_f1": rf_best_f1["f1_mean"],
                    "best_f1_param": rf_best_f1["value"],
                },
            ]
        )

        print(f"  Saved outputs in: {ds_out}")

    summary_path = out_dir / "final_summary_table.csv"
    save_rows_csv(
        summary_rows,
        ["dataset", "algorithm", "best_accuracy", "best_accuracy_param", "best_f1", "best_f1_param"],
        summary_path,
    )
    matrix_path = out_dir / "final_summary_matrix.csv"
    datasets_seen = sorted(list({r["dataset"] for r in summary_rows}))
    algorithms_seen = sorted(list({r["algorithm"] for r in summary_rows}))
    matrix_headers = ["algorithm"]
    for d in datasets_seen:
        matrix_headers.append(f"{d}_accuracy")
        matrix_headers.append(f"{d}_f1")

    matrix_rows = []
    for algo in algorithms_seen:
        row = {"algorithm": algo}
        for d in datasets_seen:
            row[f"{d}_accuracy"] = ""
            row[f"{d}_f1"] = ""
        for r in summary_rows:
            if r["algorithm"] == algo:
                row[f"{r['dataset']}_accuracy"] = r["best_accuracy"]
                row[f"{r['dataset']}_f1"] = r["best_f1"]
        matrix_rows.append(row)

    save_rows_csv(matrix_rows, matrix_headers, matrix_path)
    print(f"\nSaved final summary table: {summary_path}")
    print(f"Saved matrix summary table: {matrix_path}")


if __name__ == "__main__":
    main()
