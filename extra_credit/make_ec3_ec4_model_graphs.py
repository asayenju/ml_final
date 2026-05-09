import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
csv_path = root / 'table_results_ec3_ec4_10fold.csv'
out_dir = root / 'results'
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
with open(csv_path, 'r', newline='') as f:
    for r in csv.DictReader(f):
        rr = dict(r)
        for k in ['accuracy_mean','accuracy_std','f1_mean','f1_std']:
            rr[k] = float(rr[k])
        rows.append(rr)

order = ['rice','credit_approval','parkinsons','handwriting']
plots = [
    ('ec3_heterogeneous_ensemble', 'EC3: 3NN + 2RF (Bootstrap Voting)', 'ec3_results_by_dataset.png', '#1f77b4', '#2ca02c'),
    ('ec4_rf_error_split', 'EC4: RF with Error-Splitting', 'ec4_results_by_dataset.png', '#ff7f0e', '#d62728'),
]

x = np.arange(len(order))
width = 0.35

for model_key, title, file_name, c1, c2 in plots:
    model_rows = {r['dataset']: r for r in rows if r['model'] == model_key}
    acc = [model_rows[d]['accuracy_mean'] * 100 for d in order]
    acc_std = [model_rows[d]['accuracy_std'] * 100 for d in order]
    f1 = [model_rows[d]['f1_mean'] * 100 for d in order]
    f1_std = [model_rows[d]['f1_std'] * 100 for d in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, acc, width, yerr=acc_std, capsize=4, color=c1, alpha=0.9, label='Accuracy (%)')
    ax.bar(x + width/2, f1, width, yerr=f1_std, capsize=4, color=c2, alpha=0.9, label='F1 Macro (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Score (%)')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / file_name, dpi=150)
    plt.close(fig)

print('Saved:', out_dir / 'ec3_results_by_dataset.png')
print('Saved:', out_dir / 'ec4_results_by_dataset.png')
