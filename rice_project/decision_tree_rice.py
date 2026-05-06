import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
NUM_REPEATS = 25
MAX_DEPTH = 8


def load_csv(filename='rice.csv'):
    path = ROOT / filename
    raw_data = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                raw_data.append(row)
    if len(raw_data) == 0:
        return np.empty((0, 0)), np.empty((0,))
    if raw_data[0][0].lower().startswith('attr') or raw_data[0][-1].lower() == 'label':
        raw_data = raw_data[1:]
    raw_data = np.array(raw_data)
    X = raw_data[:, :-1].astype(float)
    y = raw_data[:, -1]
    return X, y


def get_entropy(y):
    if len(y) == 0:
        return 0
    labels, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-9))


def get_best_split(X, y, features):
    best_gain = -1.0
    best_feature = None
    best_threshold = None
    best_masks = None
    base_entropy = get_entropy(y)

    for feature in features:
        values = np.sort(np.unique(X[:, feature]))
        if len(values) <= 1:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            left_entropy = get_entropy(y[left_mask])
            right_entropy = get_entropy(y[right_mask])
            weighted_entropy = (
                left_mask.sum() / len(y) * left_entropy
                + right_mask.sum() / len(y) * right_entropy
            )
            gain = base_entropy - weighted_entropy
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_masks = (left_mask, right_mask)
        print(f"  Feature {feature}: considered {len(thresholds)} thresholds, best_gain={best_gain:.6f}")
    return best_feature, best_threshold, best_masks, best_gain


def build_tree(X, y, features, max_depth=10, depth=0):
    if len(np.unique(y)) == 1:
        return y[0]
    if len(features) == 0:
        labels_list = list(y)
        return max(set(labels_list), key=labels_list.count)
    if depth >= max_depth:
        labels_list = list(y)
        return max(set(labels_list), key=labels_list.count)

    best_feature, best_threshold, best_masks, best_gain = get_best_split(X, y, features)
    if best_feature is None or best_gain <= 1e-12:
        labels_list = list(y)
        return max(set(labels_list), key=labels_list.count)

    left_mask, right_mask = best_masks
    print(f"{'  ' * depth}Split depth={depth}: feature={best_feature}, threshold={best_threshold:.4f}, gain={best_gain:.6f}, left={left_mask.sum()}, right={right_mask.sum()}")
    remaining_features = features.copy()
    tree = {
        'feature': best_feature,
        'threshold': best_threshold,
        'left': None,
        'right': None,
    }
    tree['left'] = build_tree(X[left_mask], y[left_mask], remaining_features, max_depth, depth + 1)
    tree['right'] = build_tree(X[right_mask], y[right_mask], remaining_features, max_depth, depth + 1)
    return tree


def predict_single(x, tree):
    if not isinstance(tree, dict):
        return tree
    feature = tree['feature']
    threshold = tree['threshold']
    if x[feature] <= threshold:
        return predict_single(x, tree['left'])
    return predict_single(x, tree['right'])


def weighted_precision_score(y_true, y_pred):
    labels, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    score_sum = 0.0
    for label, count in zip(labels, counts):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        precision = tp / (tp + fp + 1e-9)
        score_sum += precision * (count / total)
    return score_sum


def weighted_recall_score(y_true, y_pred):
    labels, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    score_sum = 0.0
    for label, count in zip(labels, counts):
        tp = np.sum((y_pred == label) & (y_true == label))
        fn = np.sum((y_pred != label) & (y_true == label))
        recall = tp / (tp + fn + 1e-9)
        score_sum += recall * (count / total)
    return score_sum


def weighted_f1_score(y_true, y_pred):
    precision = weighted_precision_score(y_true, y_pred)
    recall = weighted_recall_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall + 1e-9)


def main():
    X, y = load_csv('rice.csv')
    train_accuracies = []
    test_accuracies = []
    train_f1s = []
    test_f1s = []
    train_precisions = []
    test_precisions = []
    train_recalls = []
    test_recalls = []

    print(f"Loaded data: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Running {NUM_REPEATS} random 80/20 splits with max_depth={MAX_DEPTH}")

    for i in range(NUM_REPEATS):
        print(f"Starting repeat {i + 1}/{NUM_REPEATS}")
        current_data = np.column_stack((X, y))
        np.random.shuffle(current_data)
        
        X_shuf = current_data[:, :-1].astype(float)
        y_shuf = current_data[:, -1]
        split_idx = int(0.8 * len(current_data))
        X_train, X_test = X_shuf[:split_idx], X_shuf[split_idx:]
        y_train, y_test = y_shuf[:split_idx], y_shuf[split_idx:]
        
        feature_indices = list(range(X_train.shape[1]))
        tree = build_tree(X_train, y_train, feature_indices, max_depth=MAX_DEPTH)
        
        train_preds = [predict_single(row, tree) for row in X_train]
        train_preds = np.array(train_preds)
        train_acc = np.mean(train_preds == y_train)
        train_accuracies.append(train_acc)
        train_prec = weighted_precision_score(y_train, train_preds)
        train_precisions.append(train_prec)
        train_rec = weighted_recall_score(y_train, train_preds)
        train_recalls.append(train_rec)
        train_f1 = weighted_f1_score(y_train, train_preds)
        train_f1s.append(train_f1)
        
        test_preds = [predict_single(row, tree) for row in X_test]
        test_preds = np.array(test_preds)
        test_acc = np.mean(test_preds == y_test)
        test_accuracies.append(test_acc)
        test_prec = weighted_precision_score(y_test, test_preds)
        test_precisions.append(test_prec)
        test_rec = weighted_recall_score(y_test, test_preds)
        test_recalls.append(test_rec)
        test_f1 = weighted_f1_score(y_test, test_preds)
        test_f1s.append(test_f1)

        print(f"  repeat {i + 1}/{NUM_REPEATS} done: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, train_f1={train_f1:.4f}, test_f1={test_f1:.4f}")

    print("All repeats finished")

    print(f"\n{'='*70}")
    print(f"DECISION TREE METRICS (averaged over {NUM_REPEATS} repeats)")
    print(f"{'='*70}")
    print(f"Training Accuracy:  {np.mean(train_accuracies):.4f} (SD: {np.std(train_accuracies):.4f})")
    print(f"Testing Accuracy:   {np.mean(test_accuracies):.4f} (SD: {np.std(test_accuracies):.4f})")
    print(f"Training Precision: {np.mean(train_precisions):.4f} (SD: {np.std(train_precisions):.4f})")
    print(f"Testing Precision:  {np.mean(test_precisions):.4f} (SD: {np.std(test_precisions):.4f})")
    print(f"Training Recall:    {np.mean(train_recalls):.4f} (SD: {np.std(train_recalls):.4f})")
    print(f"Testing Recall:     {np.mean(test_recalls):.4f} (SD: {np.std(test_recalls):.4f})")
    print(f"Training F1:        {np.mean(train_f1s):.4f} (SD: {np.std(train_f1s):.4f})")
    print(f"Testing F1:         {np.mean(test_f1s):.4f} (SD: {np.std(test_f1s):.4f})")
    print(f"{'='*70}\n")

    # Plotting training accuracy distribution
    plt.figure(figsize=(12, 5))
    plt.hist(train_accuracies, bins=20, range=(0.8, 1), color='skyblue', edgecolor='black')
    plt.title('Training Accuracy Distribution')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.xlim(0.8, 1.0)
    plt.savefig(ROOT / 'dt_train_accuracy_distribution.png')
    plt.close()

    # Plotting testing accuracy distribution
    plt.figure(figsize=(12, 5))
    plt.hist(test_accuracies, bins=20, range=(0.8, 1), color='salmon', edgecolor='black')
    plt.title('Testing Accuracy Distribution')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.xlim(0.8, 1.0)
    plt.savefig(ROOT / 'dt_test_accuracy_distribution.png')
    plt.close()

    # Plotting training F1 distribution
    plt.figure(figsize=(12, 5))
    plt.hist(train_f1s, bins=20, range=(0.8, 1), color='lightgreen', edgecolor='black')
    plt.title('Training F1 Distribution')
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.xlim(0.8, 1.0)
    plt.savefig(ROOT / 'dt_train_f1_distribution.png')
    plt.close()

    # Plotting testing F1 distribution
    plt.figure(figsize=(12, 5))
    plt.hist(test_f1s, bins=20, range=(0.8, 1), color='orange', edgecolor='black')
    plt.title('Testing F1 Distribution')
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.xlim(0.8, 1.0)
    plt.savefig(ROOT / 'dt_test_f1_distribution.png')
    plt.close()


if __name__ == "__main__":
    main()
