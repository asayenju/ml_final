import numpy as np


class KNNClassifier:

    def __init__(self, k=3):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if len(X) == 0:
            raise ValueError("Training set cannot be empty")

        self.X_train = X
        self.y_train = y
        return self

    def _check_is_fitted(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model is not fitted. Call fit(X, y) first.")

    def _predict_one(self, x_new):
        # Euclidean Distance
        diff = self.X_train - x_new
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        k = min(self.k, len(self.y_train))
        k_indices = np.argsort(distances)[:k]
        k_labels = self.y_train[k_indices]

        labels, counts = np.unique(k_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, X):
        self._check_is_fitted()

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")

        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        if len(y) != len(y_pred):
            raise ValueError("X and y must have the same number of samples")
        return np.mean(y == y_pred)
