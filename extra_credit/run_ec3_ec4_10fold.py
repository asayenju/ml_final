import csv
from pathlib import Path
import sys

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library import HeterogeneousBootstrapEnsembleEC3, RandomForestClassifierErrorSplitScratch

OUT_DIR = ROOT / "extra_credit" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_FOLDS = 10


def load_csv_dataset(path, label_col=-1):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)

    header = rows[0]
    data_rows = rows[1:]
    data = np.array(data_rows, dtype=object)

    if isinstance(label_col, str):
        idx = header.index(label_col)
    else:
        idx = label_col if label_col >= 0 else data.shape[1] + label_col

    feature_idx = [i for i in range(data.shape[1]) if i != idx]
    X = data[:, feature_idx]
    y = data[:, idx]
    return X, y


def evaluate_model(model, X, y, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        print(f"    fold {fold}/{k}: acc={acc:.4f}, f1={f1:.4f}", flush=True)

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }


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
    return set(numeric_cols)


def main():
    dataset_specs = [
        ("rice", ROOT / "rice.csv", -1),
        ("credit_approval", ROOT / "credit_approval.csv", -1),
        ("parkinsons", ROOT / "parkinsons.csv", "Diagnosis"),
    ]

    results = []

    for name, path, label_col in dataset_specs:
        print(f"\n=== Dataset: {name} ===", flush=True)
        X, y = load_csv_dataset(path, label_col)

        print("  Running EC3 HeterogeneousBootstrapEnsembleEC3...", flush=True)
        ec3 = HeterogeneousBootstrapEnsembleEC3(
            nn_architectures=[[10], [8, 4], [12, 6]],
            n_rf_members=2,
            rf_params={"n_trees": 7, "max_depth": 8, "min_size": 4, "min_gain": 1e-4},
            nn_max_iterations=10,
            random_state=RANDOM_STATE,
        )
        ec3_metrics = evaluate_model(ec3, X, y, k=K_FOLDS)
        results.append({"dataset": name, "model": "ec3_heterogeneous_ensemble", **ec3_metrics})

        print("  Running EC4 RF Error-Split variant...", flush=True)
        ec4 = RandomForestClassifierErrorSplitScratch(
            n_trees=12,
            max_depth=10,
            min_size=3,
            min_gain=1e-4,
            numeric_cols=infer_numeric_cols(X),
            random_state=RANDOM_STATE,
        )
        ec4_metrics = evaluate_model(ec4, X, y, k=K_FOLDS)
        results.append({"dataset": name, "model": "ec4_rf_error_split", **ec4_metrics})

    print("\n=== Dataset: handwriting (sklearn digits) ===", flush=True)
    Xh, yh = datasets.load_digits(return_X_y=True)
    Xh = Xh.astype(object)

    print("  Running EC3 HeterogeneousBootstrapEnsembleEC3...", flush=True)
    ec3 = HeterogeneousBootstrapEnsembleEC3(
        nn_architectures=[[12], [10, 5], [16, 8]],
        n_rf_members=2,
        rf_params={"n_trees": 9, "max_depth": 9, "min_size": 4, "min_gain": 1e-4},
        nn_max_iterations=10,
        random_state=RANDOM_STATE,
    )
    ec3_metrics = evaluate_model(ec3, Xh, yh, k=K_FOLDS)
    results.append({"dataset": "handwriting", "model": "ec3_heterogeneous_ensemble", **ec3_metrics})

    print("  Running EC4 RF Error-Split variant...", flush=True)
    ec4 = RandomForestClassifierErrorSplitScratch(
        n_trees=14,
        max_depth=12,
        min_size=3,
        min_gain=1e-4,
        numeric_cols=infer_numeric_cols(Xh),
        random_state=RANDOM_STATE,
    )
    ec4_metrics = evaluate_model(ec4, Xh, yh, k=K_FOLDS)
    results.append({"dataset": "handwriting", "model": "ec4_rf_error_split", **ec4_metrics})

    out_path = ROOT / "extra_credit" / "table_results_ec3_ec4_10fold.csv"
    headers = ["dataset", "model", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {out_path}")
    for row in results:
        print(
            f"{row['dataset']:>15} | {row['model']:<33} | "
            f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
            f"f1={row['f1_mean']:.4f}±{row['f1_std']:.4f}"
        )


if __name__ == "__main__":
    main()
