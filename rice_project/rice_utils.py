import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve

sns.set(style='whitegrid', palette='muted', font_scale=1.1)

ROOT = Path(__file__).resolve().parent


def load_rice_data(filename: str = 'rice.csv'):
    path = ROOT / filename
    df = pd.read_csv(path)
    X = df.drop(columns=['label']).values
    y = LabelEncoder().fit_transform(df['label'])
    return X, y


def evaluate_model_cv(estimator, X, y, cv_folds: int = 10):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracy = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    f1 = cross_val_score(estimator, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
    return {
        'accuracy_mean': float(np.mean(accuracy)),
        'accuracy_std': float(np.std(accuracy)),
        'f1_mean': float(np.mean(f1)),
        'f1_std': float(np.std(f1)),
    }


def evaluate_hyperparameters(build_model, param_grid, X, y, cv_folds: int = 10):
    rows = []
    for params in param_grid:
        estimator = build_model(**params)
        stats = evaluate_model_cv(estimator, X, y, cv_folds=cv_folds)
        row = {**params, **stats}
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(results, filename: str):
    path = ROOT / filename
    results.to_csv(path, index=False)
    return path


def plot_hyperparam_results(results, x_col: str, output_path: str, title: str):
    path = ROOT / output_path
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=results, x=x_col, y='accuracy_mean', marker='o', label='Accuracy', ax=ax)
    sns.lineplot(data=results, x=x_col, y='f1_mean', marker='o', label='F1-score', ax=ax)
    ax.fill_between(results[x_col], results['accuracy_mean'] - results['accuracy_std'], results['accuracy_mean'] + results['accuracy_std'], alpha=0.15)
    ax.fill_between(results[x_col], results['f1_mean'] - results['f1_std'], results['f1_mean'] + results['f1_std'], alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_learning_curve(estimator, X, y, output_path: str, title: str, cv_folds: int = 10):
    path = ROOT / output_path
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.2, 1.0, 5)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring='f1_weighted',
        train_sizes=train_sizes,
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes_abs, train_mean, marker='o', label='Train F1-score')
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.plot(train_sizes_abs, test_mean, marker='s', label='Validation F1-score')
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel('Training examples')
    ax.set_ylabel('F1-score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
