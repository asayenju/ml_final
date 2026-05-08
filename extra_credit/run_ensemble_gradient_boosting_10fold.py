import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from library import AdaBoostSAMMEScratch, GradientBoostingClassifierScratch

ROOT = Path(__file__).resolve().parent
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
    accs = []
    f1s = []

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


def save_plots(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets_order = ["rice", "credit_approval", "parkinsons", "handwriting"]
    models = ["ensemble_adaboost", "gradient_boosting"]
    labels = {
        "ensemble_adaboost": "Ensemble (AdaBoost)",
        "gradient_boosting": "Gradient Boosting",
    }
    colors = {
        "ensemble_adaboost": "#1f77b4",
        "gradient_boosting": "#ff7f0e",
    }

    metric_settings = [
        ("accuracy_mean", "accuracy_std", "Accuracy", "accuracy_comparison_10fold.png"),
        ("f1_mean", "f1_std", "F1 (macro)", "f1_comparison_10fold.png"),
    ]

    x = np.arange(len(datasets_order))
    width = 0.35

    for mean_key, std_key, y_label, filename in metric_settings:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, model_name in enumerate(models):
            means = []
            stds = []
            for ds in datasets_order:
                row = next(r for r in results if r["dataset"] == ds and r["model"] == model_name)
                means.append(row[mean_key])
                stds.append(row[std_key])
            positions = x + (i - 0.5) * width
            ax.bar(
                positions,
                means,
                width=width,
                yerr=stds,
                capsize=4,
                label=labels[model_name],
                color=colors[model_name],
                alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(datasets_order)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} Comparison ({K_FOLDS}-Fold CV)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)


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

        print("  Running AdaBoostSAMMEScratch...", flush=True)
        ada = AdaBoostSAMMEScratch(n_estimators=12, random_state=RANDOM_STATE)
        ada_metrics = evaluate_model(ada, X, y, k=K_FOLDS)
        results.append({"dataset": name, "model": "ensemble_adaboost", **ada_metrics})

        print("  Running GradientBoostingClassifierScratch...", flush=True)
        gb = GradientBoostingClassifierScratch(n_estimators=15, learning_rate=0.1, random_state=RANDOM_STATE)
        gb_metrics = evaluate_model(gb, X, y, k=K_FOLDS)
        results.append({"dataset": name, "model": "gradient_boosting", **gb_metrics})

    print("\n=== Dataset: handwriting (sklearn digits) ===", flush=True)
    Xh, yh = datasets.load_digits(return_X_y=True)
    Xh = Xh.astype(object)

    print("  Running AdaBoostSAMMEScratch...", flush=True)
    ada = AdaBoostSAMMEScratch(n_estimators=15, random_state=RANDOM_STATE)
    ada_metrics = evaluate_model(ada, Xh, yh, k=K_FOLDS)
    results.append({"dataset": "handwriting", "model": "ensemble_adaboost", **ada_metrics})

    print("  Running GradientBoostingClassifierScratch...", flush=True)
    gb = GradientBoostingClassifierScratch(n_estimators=18, learning_rate=0.1, random_state=RANDOM_STATE)
    gb_metrics = evaluate_model(gb, Xh, yh, k=K_FOLDS)
    results.append({"dataset": "handwriting", "model": "gradient_boosting", **gb_metrics})

    out_path = ROOT / "table_results_ensemble_gb_10fold.csv"
    headers = ["dataset", "model", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    plot_dir = ROOT / "results" / "ensemble_gb"
    save_plots(results, plot_dir)

    print(f"\nSaved results to: {out_path}")
    print(f"Saved plots in: {plot_dir}")
    for row in results:
        print(
            f"{row['dataset']:>15} | {row['model']:<20} | "
            f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
            f"f1={row['f1_mean']:.4f}±{row['f1_std']:.4f}"
        )


if __name__ == "__main__":
    main()
