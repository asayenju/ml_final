import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "table_results_ensemble_gb_10fold.csv"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
with open(CSV_PATH, "r", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r2 = dict(r)
        for k in ["accuracy_mean", "accuracy_std", "f1_mean", "f1_std"]:
            r2[k] = float(r2[k])
        rows.append(r2)

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
    ("accuracy_mean", "accuracy_std", "Accuracy (%)", "accuracy_comparison_10fold.png"),
    ("f1_mean", "f1_std", "F1 Macro (%)", "f1_comparison_10fold.png"),
]

x = np.arange(len(datasets_order))
width = 0.35

for mean_key, std_key, y_label, filename in metric_settings:
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, m in enumerate(models):
        means = []
        stds = []
        for ds in datasets_order:
            row = next(rr for rr in rows if rr["dataset"] == ds and rr["model"] == m)
            means.append(row[mean_key] * 100.0)
            stds.append(row[std_key] * 100.0)

        pos = x + (i - 0.5) * width
        ax.bar(
            pos,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=labels[m],
            color=colors[m],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets_order)
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} Comparison (10-Fold CV)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)

print(f"Saved plots to: {OUT_DIR}")
