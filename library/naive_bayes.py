import numpy as np


class MultinomialNaiveBayesScratch:

    def __init__(self, alpha=1.0):
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = float(alpha)

        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.n_features_ = None

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
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative feature values")

        self.n_features_ = X.shape[1]
        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.log(class_counts / class_counts.sum())
        self.feature_log_prob_ = np.zeros((n_classes, self.n_features_), dtype=float)

        for idx, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            feature_count = X_c.sum(axis=0)
            smoothed_fc = feature_count + self.alpha
            smoothed_total = smoothed_fc.sum()
            self.feature_log_prob_[idx, :] = np.log(smoothed_fc / smoothed_total)

        return self

    def _check_is_fitted(self):
        if self.classes_ is None or self.feature_log_prob_ is None:
            raise ValueError("Model is not fitted. Call fit(X, y) first.")

    def _joint_log_likelihood(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_log_proba(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if X.shape[1] != self.n_features_:
            raise ValueError("X must have the same number of features as training data")
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative feature values")

        jll = self._joint_log_likelihood(X)


        max_log = np.max(jll, axis=1, keepdims=True)
        log_prob_x = max_log + np.log(np.sum(np.exp(jll - max_log), axis=1, keepdims=True))
        return jll - log_prob_x

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        jll = self._joint_log_likelihood(X)
        class_indices = np.argmax(jll, axis=1)
        return self.classes_[class_indices]

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        if len(y) != len(y_pred):
            raise ValueError("X and y must have the same number of samples")
        return np.mean(y == y_pred)
