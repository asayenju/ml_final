import numpy as np

# Step-by-step usage:
# 1) Import: from library import DecisionTreeClassifierScratch
# 2) Prepare data:
#    - X_train shape = (n_samples, n_features), categorical/string features
#    - y_train shape = (n_samples,)
# 3) Create model: model = DecisionTreeClassifierScratch()
# 4) Train: model.fit(X_train, y_train)
# 5) Predict labels: y_pred = model.predict(X_test)
# 6) Evaluate accuracy: acc = model.score(X_test, y_test)
# 7) For unseen category values at inference, the model uses node-level fallback labels.


class DecisionTreeClassifierScratch:
    """A simple ID3-style Decision Tree classifier for categorical features."""

    def __init__(self):
        self.tree_ = None
        self.default_label_ = None
        self.n_features_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if len(X) == 0:
            raise ValueError("Training set cannot be empty")

        self.n_features_ = X.shape[1]
        self.default_label_ = self._majority_label(y)
        features = list(range(self.n_features_))
        self.tree_ = self._build_tree(X, y, features)
        return self

    def _check_is_fitted(self):
        if self.tree_ is None:
            raise ValueError("Model is not fitted. Call fit(X, y) first.")

    @staticmethod
    def _entropy(y):
        labels, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-12))

    @staticmethod
    def _majority_label(y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _information_gain(self, X, y, feature_index):
        parent_entropy = self._entropy(y)

        values, counts = np.unique(X[:, feature_index], return_counts=True)
        weighted_entropy = 0.0

        for val, count in zip(values, counts):
            mask = X[:, feature_index] == val
            subset_y = y[mask]
            child_entropy = self._entropy(subset_y)
            weighted_entropy += (count / len(y)) * child_entropy

        return parent_entropy - weighted_entropy

    def _build_tree(self, X, y, features):
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            return unique_labels[0]

        if len(features) == 0:
            return self._majority_label(y)

        gains = [self._information_gain(X, y, f) for f in features]
        best_feature = features[int(np.argmax(gains))]

        node = {
            "feature": best_feature,
            "children": {},
            "fallback": self._majority_label(y),
        }

        remaining_features = [f for f in features if f != best_feature]
        for val in np.unique(X[:, best_feature]):
            mask = X[:, best_feature] == val
            if np.any(mask):
                node["children"][val] = self._build_tree(X[mask], y[mask], remaining_features)
            else:
                node["children"][val] = node["fallback"]

        return node

    def _predict_one(self, x):
        node = self.tree_
        while isinstance(node, dict):
            feature_index = node["feature"]
            feature_val = x[feature_index]
            if feature_val in node["children"]:
                node = node["children"][feature_val]
            else:
                return node["fallback"]
        return node

    def predict(self, X):
        self._check_is_fitted()

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if X.shape[1] != self.n_features_:
            raise ValueError("X must have the same number of features as training data")

        return np.array([self._predict_one(row) for row in X])

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        if len(y) != len(y_pred):
            raise ValueError("X and y must have the same number of samples")
        return np.mean(y == y_pred)
