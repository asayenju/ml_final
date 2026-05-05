import csv
import numpy as np
import matplotlib.pyplot as plt


def load_csv(filename='rice.csv'):
    raw_data = []
    with open(filename, 'r', newline='') as f:
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


def predict_single(x_new, X_train, y_train, k):
    distances = []
    for x_train in X_train:
        diff = x_new - x_train
        dist = np.sqrt(np.sum(diff**2))
        distances.append(dist)
    distances = np.array(distances)
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[k_indices]
    labels_list = list(k_nearest_labels)
    prediction = max(set(labels_list), key=labels_list.count)
    return prediction


def predict_batch(X_to_predict, X_train, y_train, k):
    predictions = []
    for x in X_to_predict:
        label = predict_single(x, X_train, y_train, k)
        predictions.append(label)
    return np.array(predictions)


def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def weighted_f1_score(y_true, y_pred):
    labels, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    score_sum = 0.0
    for label, count in zip(labels, counts):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        score_sum += f1 * (count / total)
    return score_sum


def main():
    filename = 'rice.csv'
    raw_data = []
    try:
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    raw_data.append(row)
        raw_data = np.array(raw_data)
    except FileNotFoundError:
        print('CSV not found.')
        return

    k_values = list(range(1, 52, 2))
    num_runs = 20

    avg_train_accs, std_train_accs = [], []
    avg_test_accs, std_test_accs = [], []
    avg_train_f1s, std_train_f1s = [], []
    avg_test_f1s, std_test_f1s = [], []

    print('Starting k-NN evaluation on rice.csv')
    for k in k_values:
        run_train_accs = []
        run_test_accs = []
        run_train_f1s = []
        run_test_f1s = []

        for run in range(num_runs):
            current_data = raw_data.copy()
            np.random.shuffle(current_data)

            X = current_data[:, :-1].astype(float)
            y = current_data[:, -1]
            split_idx = int(0.8 * len(current_data))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            train_preds = predict_batch(X_train, X_train, y_train, k)
            run_train_accs.append(calculate_accuracy(y_train, train_preds))
            run_train_f1s.append(weighted_f1_score(y_train, train_preds))

            test_preds = predict_batch(X_test, X_train, y_train, k)
            run_test_accs.append(calculate_accuracy(y_test, test_preds))
            run_test_f1s.append(weighted_f1_score(y_test, test_preds))

            if (run + 1) % 5 == 0:
                print(f'  k={k} run {run + 1}/{num_runs} completed')

        avg_train_accs.append(np.mean(run_train_accs))
        std_train_accs.append(np.std(run_train_accs))
        avg_test_accs.append(np.mean(run_test_accs))
        std_test_accs.append(np.std(run_test_accs))
        avg_train_f1s.append(np.mean(run_train_f1s))
        std_train_f1s.append(np.std(run_train_f1s))
        avg_test_f1s.append(np.mean(run_test_f1s))
        std_test_f1s.append(np.std(run_test_f1s))
        print(f'k={k} completed.')

    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, avg_train_accs, yerr=std_train_accs, fmt='-o', color='blue', ecolor='lightblue', capsize=3, label='Train Accuracy')
    plt.title('Training Accuracy vs k')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('knn_train_accuracy.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, avg_test_accs, yerr=std_test_accs, fmt='-s', color='purple', ecolor='thistle', capsize=3, label='Test Accuracy')
    plt.title('Testing Accuracy vs k')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('knn_test_accuracy.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, avg_train_f1s, yerr=std_train_f1s, fmt='-o', color='green', ecolor='lightgreen', capsize=3, label='Train F1')
    plt.title('Training F1 vs k')
    plt.xlabel('Value of k')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.legend()
    plt.savefig('knn_train_f1.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, avg_test_f1s, yerr=std_test_f1s, fmt='-s', color='red', ecolor='pink', capsize=3, label='Test F1')
    plt.title('Testing F1 vs k')
    plt.xlabel('Value of k')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.legend()
    plt.savefig('knn_test_f1.png')
    plt.show()


if __name__ == '__main__':
    main()
