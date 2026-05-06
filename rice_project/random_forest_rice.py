import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent


def get_hyperparameters():
    max_depth = 10
    min_size = 3
    min_gain = 1e-4
    ntree_values = [1, 5, 10, 20, 30, 40, 50]
    k_folds = 10
    return max_depth, min_size, min_gain, ntree_values, k_folds


MAX_DEPTH, MIN_SIZE, MIN_GAIN, NTREE_VALUES, K_FOLDS = get_hyperparameters()


def load_csv(filename='rice.csv'):
    path = ROOT / filename
    raw_data = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                raw_data.append(row)
    raw_data = np.array(raw_data)
    label_idx = len(header) - 1
    for i, col_name in enumerate(header):
        if col_name.lower() == 'label':
            label_idx = i
            break
    feature_indices = [i for i in range(len(header)) if i != label_idx]
    X = raw_data[:, feature_indices].astype(float)
    y = raw_data[:, label_idx]
    return X, y


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))


def best_split_numerical(X, y, col):
    col_vals = X[:, col].astype(float)
    order = np.argsort(col_vals)
    col_sorted = col_vals[order]
    y_sorted = y[order]
    n = len(y)
    parent_entropy = entropy(y)

    best_gain = -1.0
    best_thresh = None

    for sp in np.where(np.diff(col_sorted) > 0)[0]:
        left_mask = col_vals <= (col_sorted[sp] + col_sorted[sp + 1]) / 2.0
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue

        left_entropy = entropy(y[left_mask])
        right_entropy = entropy(y[right_mask])
        weighted_entropy = (
            left_mask.sum() / n * left_entropy
            + right_mask.sum() / n * right_entropy
        )
        gain = parent_entropy - weighted_entropy
        if gain > best_gain:
            best_gain = gain
            best_thresh = (col_sorted[sp] + col_sorted[sp + 1]) / 2.0

    return best_thresh, best_gain


def majority_vote(y):
    labels, counts = np.unique(y, return_counts=True)
    return labels[np.argmax(counts)]


def build_tree(X, y, features, max_depth=None, min_samples_leaf=1, min_gain=1e-4, depth=0):
    if len(np.unique(y)) == 1:
        return y[0]
    if len(features) == 0 or len(y) < 2 * min_samples_leaf or (max_depth is not None and depth >= max_depth):
        return majority_vote(y)

    best_feature = None
    best_thresh = None
    best_gain = -1.0

    for f in features:
        thresh, gain = best_split_numerical(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
            best_thresh = thresh

    if best_feature is None or best_gain < min_gain:
        return majority_vote(y)

    left_mask = X[:, best_feature].astype(float) <= best_thresh
    right_mask = ~left_mask
    if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
        return majority_vote(y)

    node = {
        'feature': best_feature,
        'threshold': best_thresh,
        'left': None,
        'right': None,
    }
    node['left'] = build_tree(
        X[left_mask], y[left_mask], features, max_depth, min_samples_leaf, min_gain, depth + 1
    )
    node['right'] = build_tree(
        X[right_mask], y[right_mask], features, max_depth, min_samples_leaf, min_gain, depth + 1
    )
    return node


def predict_one(x, tree):
    if not isinstance(tree, dict):
        return tree
    if float(x[tree['feature']]) <= tree['threshold']:
        return predict_one(x, tree['left'])
    return predict_one(x, tree['right'])


def predict_batch(X, tree):
    return np.array([predict_one(x, tree) for x in X])


def build_forest(X, y, n_trees, max_depth, min_samples_leaf, min_gain, feature_subset_size):
    trees = []
    n = len(y)
    for _ in range(n_trees):
        indices = np.random.choice(n, size=n, replace=True)
        sample_X = X[indices]
        sample_y = y[indices]
        features = list(np.random.choice(X.shape[1], size=feature_subset_size, replace=False))
        tree = build_tree(sample_X, sample_y, features, max_depth, min_samples_leaf, min_gain)
        trees.append(tree)
    return trees


def forest_predict(X, trees):
    all_preds = np.array([[predict_one(x, tree) for tree in trees] for x in X])
    return np.array([max(set(row), key=list(row).count) for row in all_preds])


def calculate_metrics(y_true, y_pred):
    labels, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    accuracy = np.mean(y_true == y_pred)
    precision = 0.0
    recall = 0.0
    for label, count in zip(labels, counts):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        if tp + fp > 0:
            precision += (tp / (tp + fp)) * (count / total)
        if tp + fn > 0:
            recall += (tp / (tp + fn)) * (count / total)
    f1 = 2 * precision * recall / (precision + recall + 1e-12) if precision + recall > 0 else 0.0
    return accuracy, precision, recall, f1


def evaluate_random_forest(X, y, ntree_values, max_depth, min_samples_leaf, min_gain, k_folds):
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []
    feature_subset_size = max(1, int(np.sqrt(X.shape[1])))

    for ntree in ntree_values:
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        print(f'Evaluating ntree={ntree}...')

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            forest = build_forest(
                X_train,
                y_train,
                ntree,
                max_depth,
                min_samples_leaf,
                min_gain,
                feature_subset_size,
            )
            y_pred = forest_predict(X_test, forest)
            accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        mean_acc = np.mean(accuracies)
        mean_prec = np.mean(precisions)
        mean_rec = np.mean(recalls)
        mean_f1 = np.mean(f1_scores)
        results.append({
            'n_estimators': ntree,
            'accuracy_mean': mean_acc,
            'accuracy_std': np.std(accuracies),
            'precision_mean': mean_prec,
            'precision_std': np.std(precisions),
            'recall_mean': mean_rec,
            'recall_std': np.std(recalls),
            'f1_mean': mean_f1,
            'f1_std': np.std(f1_scores),
        })
        print(f'  accuracy={mean_acc:.4f}, precision={mean_prec:.4f}, recall={mean_rec:.4f}, f1={mean_f1:.4f}')

    return np.array(results, dtype=object)


def save_results(results, filename):
    path = ROOT / filename
    import csv as csv_lib
    with open(path, 'w', newline='') as f:
        writer = csv_lib.writer(f)
        writer.writerow(list(results[0].keys()))
        for row in results:
            writer.writerow([row[k] for k in row])
    return path


def plot_metric(ntree_values, values, ylabel, filename, title, color):
    path = ROOT / filename
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ntree_values, values, marker='o', color=color)
    ax.set_title(title)
    ax.set_xlabel('Number of Trees (ntree)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ntree_values)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    X, y = load_csv('rice.csv')
    print('Running scratch random forest on rice dataset...')
    results = evaluate_random_forest(X, y, NTREE_VALUES, MAX_DEPTH, MIN_SIZE, MIN_GAIN, K_FOLDS)
    columns = [
        'n_estimators',
        'accuracy_mean',
        'accuracy_std',
        'precision_mean',
        'precision_std',
        'recall_mean',
        'recall_std',
        'f1_mean',
        'f1_std',
    ]
    results_df = np.array([[row[col] for col in columns] for row in results], dtype=object)
    header = columns
    save_results(results, 'random_forest_hyperparam_results.csv')
    print('\nHyperparameter results:')
    print(header)
    for row in results:
        print([row[col] for col in header])

    ntree_values = [row['n_estimators'] for row in results]
    plot_metric(ntree_values, [row['accuracy_mean'] for row in results], 'Accuracy', 'rf_accuracy_vs_ntree.png', 'Random Forest Accuracy vs ntree', '#1f77b4')
    plot_metric(ntree_values, [row['precision_mean'] for row in results], 'Precision', 'rf_precision_vs_ntree.png', 'Random Forest Precision vs ntree', '#ff7f0e')
    plot_metric(ntree_values, [row['recall_mean'] for row in results], 'Recall', 'rf_recall_vs_ntree.png', 'Random Forest Recall vs ntree', '#2ca02c')
    plot_metric(ntree_values, [row['f1_mean'] for row in results], 'F1 Score', 'rf_f1_vs_ntree.png', 'Random Forest F1 Score vs ntree', '#d62728')

    print('\nSaved random_forest_hyperparam_results.csv and 4 metric plots.')


if __name__ == '__main__':
    main()
