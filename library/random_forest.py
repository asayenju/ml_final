import numpy as np


class RandomForestClassifierScratch:

    def __init__(
        self,
        n_trees=10,
        max_depth=10,
        min_size=3,
        min_gain=1e-4,
        numeric_cols=None,
        max_features=None,
        random_state=None,
    ):
        if n_trees < 1:
            raise ValueError("n_trees must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if min_size < 1:
            raise ValueError("min_size must be >= 1")

        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.min_size = int(min_size)
        self.min_gain = float(min_gain)
        self.numeric_cols = set(numeric_cols or [])
        self.max_features = max_features
        self.random_state = random_state

        self.trees_ = []
        self.n_features_ = None
        self.default_label_ = None

        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
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
        self.default_label_ = self._majority_vote(y)

        if self.max_features is None:
            m = max(1, int(np.sqrt(self.n_features_)))
        else:
            m = int(self.max_features)
            if m < 1:
                raise ValueError("max_features must be >= 1")
        m = min(m, self.n_features_)

        self.trees_ = []
        all_features = list(range(self.n_features_))

        for _ in range(self.n_trees):
            boot_idx = self._rng.choice(len(y), size=len(y), replace=True)
            tree = self._build_tree(X[boot_idx], y[boot_idx], all_features, m, depth=0)
            self.trees_.append(tree)

        return self

    def _check_is_fitted(self):
        if not self.trees_:
            raise ValueError("Model is not fitted. Call fit(X, y) first.")

    @staticmethod
    def _entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-12))

    @staticmethod
    def _majority_vote(y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _information_gain_categorical(self, X, y, col):
        parent_entropy = self._entropy(y)
        values, counts = np.unique(X[:, col], return_counts=True)
        weighted_entropy = 0.0

        for val, count in zip(values, counts):
            child_y = y[X[:, col] == val]
            weighted_entropy += (count / len(y)) * self._entropy(child_y)

        return parent_entropy - weighted_entropy

    def _best_split_numerical(self, X, y, col):
        col_vals = X[:, col].astype(float)
        parent_entropy = self._entropy(y)
        n = len(y)

        order = np.argsort(col_vals)
        col_sorted = col_vals[order]

        best_gain = -1.0
        best_thresh = None

        for sp in np.where(np.diff(col_sorted) > 0)[0]:
            thresh = (col_sorted[sp] + col_sorted[sp + 1]) / 2.0
            left_mask = col_vals <= thresh
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            left_entropy = self._entropy(y[left_mask])
            right_entropy = self._entropy(y[right_mask])
            weighted = (left_mask.sum() / n) * left_entropy + (right_mask.sum() / n) * right_entropy
            gain = parent_entropy - weighted

            if gain > best_gain:
                best_gain = gain
                best_thresh = thresh

        return best_thresh, best_gain

    def _build_tree(self, X, y, all_features, m, depth):
        if len(np.unique(y)) == 1:
            return y[0]
        if len(all_features) == 0 or len(y) < self.min_size or depth >= self.max_depth:
            return self._majority_vote(y)

        selected = list(self._rng.choice(all_features, size=min(m, len(all_features)), replace=False))

        best_feat = None
        best_gain = -1.0
        best_thresh = None

        for f in selected:
            if f in self.numeric_cols:
                thresh, gain = self._best_split_numerical(X, y, f)
            else:
                gain = self._information_gain_categorical(X, y, f)
                thresh = None

            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thresh = thresh

        if best_feat is None or best_gain < self.min_gain:
            return self._majority_vote(y)

        remaining = [f for f in all_features if f != best_feat]

        if best_feat in self.numeric_cols:
            vals = X[:, best_feat].astype(float)
            left_mask = vals <= best_thresh
            right_mask = ~left_mask

            left = self._build_tree(X[left_mask], y[left_mask], remaining, m, depth + 1) if left_mask.sum() > 0 else self._majority_vote(y)
            right = self._build_tree(X[right_mask], y[right_mask], remaining, m, depth + 1) if right_mask.sum() > 0 else self._majority_vote(y)

            return {
                "feature": best_feat,
                "type": "num",
                "threshold": best_thresh,
                "left": left,
                "right": right,
            }

        node = {
            "feature": best_feat,
            "type": "cat",
            "branches": {},
            "default": self._majority_vote(y),
        }

        for val in np.unique(X[:, best_feat]):
            mask = X[:, best_feat] == val
            node["branches"][val] = self._build_tree(X[mask], y[mask], remaining, m, depth + 1)

        return node

    def _predict_one_tree(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        f = tree["feature"]
        if tree["type"] == "num":
            return self._predict_one_tree(x, tree["left"] if float(x[f]) <= tree["threshold"] else tree["right"])

        return self._predict_one_tree(x, tree["branches"].get(x[f], tree["default"]))

    def _forest_vote(self, row_preds):
        labels, counts = np.unique(row_preds, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, X):
        self._check_is_fitted()

        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if X.shape[1] != self.n_features_:
            raise ValueError("X must have the same number of features as training data")

        all_preds = np.array([[self._predict_one_tree(x, t) for t in self.trees_] for x in X])
        return np.array([self._forest_vote(row) for row in all_preds])

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        if len(y) != len(y_pred):
            raise ValueError("X and y must have the same number of samples")
        return np.mean(y == y_pred)
