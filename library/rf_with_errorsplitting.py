import numpy as np


class RandomForestClassifierErrorSplitScratch:
    """
    Random Forest variant using classification-error reduction
    as split criterion (instead of entropy/information gain or gini).
    """

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

    @staticmethod
    def _majority_vote(y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    @staticmethod
    def _classification_error(y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.max(probs)

    def _error_reduction_categorical(self, X, y, col):
        parent_err = self._classification_error(y)
        values, counts = np.unique(X[:, col], return_counts=True)
        weighted_err = 0.0
        for val, count in zip(values, counts):
            child_y = y[X[:, col] == val]
            weighted_err += (count / len(y)) * self._classification_error(child_y)
        return parent_err - weighted_err

    def _best_split_numerical(self, X, y, col):
        col_vals = X[:, col].astype(float)
        parent_err = self._classification_error(y)
        n = len(y)

        order = np.argsort(col_vals)
        sorted_vals = col_vals[order]

        best_gain = -1.0
        best_thresh = None
        for sp in np.where(np.diff(sorted_vals) > 0)[0]:
            thresh = (sorted_vals[sp] + sorted_vals[sp + 1]) / 2.0
            left = col_vals <= thresh
            right = ~left
            if left.sum() == 0 or right.sum() == 0:
                continue
            weighted_err = (left.sum() / n) * self._classification_error(y[left]) + (right.sum() / n) * self._classification_error(y[right])
            gain = parent_err - weighted_err
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
        best_feat, best_gain, best_thresh = None, -1.0, None

        for f in selected:
            if f in self.numeric_cols:
                thresh, gain = self._best_split_numerical(X, y, f)
            else:
                thresh = None
                gain = self._error_reduction_categorical(X, y, f)
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
            return {"feature": best_feat, "type": "num", "threshold": best_thresh, "left": left, "right": right}

        node = {"feature": best_feat, "type": "cat", "branches": {}, "default": self._majority_vote(y)}
        for val in np.unique(X[:, best_feat]):
            mask = X[:, best_feat] == val
            node["branches"][val] = self._build_tree(X[mask], y[mask], remaining, m, depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1D and match X rows")

        self.n_features_ = X.shape[1]
        self.default_label_ = self._majority_vote(y)
        if self.max_features is None:
            m = max(1, int(np.sqrt(self.n_features_)))
        else:
            m = max(1, int(self.max_features))
        m = min(m, self.n_features_)

        self.trees_ = []
        all_features = list(range(self.n_features_))
        for _ in range(self.n_trees):
            idx = self._rng.choice(len(y), size=len(y), replace=True)
            tree = self._build_tree(X[idx], y[idx], all_features, m, depth=0)
            self.trees_.append(tree)
        return self

    def _predict_one_tree(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        f = tree["feature"]
        if tree["type"] == "num":
            nxt = tree["left"] if float(x[f]) <= tree["threshold"] else tree["right"]
            return self._predict_one_tree(x, nxt)
        return self._predict_one_tree(x, tree["branches"].get(x[f], tree["default"]))

    def predict(self, X):
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        all_preds = np.array([[self._predict_one_tree(x, t) for t in self.trees_] for x in X])
        out = []
        for row in all_preds:
            labels, counts = np.unique(row, return_counts=True)
            out.append(labels[np.argmax(counts)])
        return np.array(out)

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)
