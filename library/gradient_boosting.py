import numpy as np


class _SimpleOneHotEncoder:
    def __init__(self):
        self.is_numeric_ = None
        self.numeric_means_ = {}
        self.categories_ = {}

    def _is_numeric_column(self, col):
        for v in col:
            s = str(v).strip()
            if s == "":
                continue
            try:
                float(s)
            except ValueError:
                return False
        return True

    def fit(self, X):
        X = np.asarray(X, dtype=object)
        self.is_numeric_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            is_num = self._is_numeric_column(col)
            self.is_numeric_.append(is_num)
            if is_num:
                vals = np.array([float(v) if str(v).strip() != "" else np.nan for v in col], dtype=float)
                mean = np.nanmean(vals) if np.any(~np.isnan(vals)) else 0.0
                self.numeric_means_[j] = float(mean)
            else:
                cats = sorted(list(set(str(v).strip() if str(v).strip() != "" else "MISSING" for v in col)))
                self.categories_[j] = cats
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        out_parts = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if self.is_numeric_[j]:
                mean = self.numeric_means_[j]
                vals = np.array([float(v) if str(v).strip() != "" else np.nan for v in col], dtype=float)
                vals = np.where(np.isnan(vals), mean, vals)
                out_parts.append(vals.reshape(-1, 1))
            else:
                cats = self.categories_[j]
                idx = {c: i for i, c in enumerate(cats)}
                block = np.zeros((X.shape[0], len(cats)), dtype=float)
                for i, v in enumerate(col):
                    key = str(v).strip() if str(v).strip() != "" else "MISSING"
                    if key in idx:
                        block[i, idx[key]] = 1.0
                out_parts.append(block)
        return np.hstack(out_parts) if out_parts else np.empty((X.shape[0], 0), dtype=float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class _RegressionStump:
    """Depth-1 regression tree trained with squared error."""

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = 0.0
        self.right_value = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_loss = float("inf")

        for j in range(n_features):
            vals = X[:, j]
            uniq = np.unique(vals)
            if len(uniq) == 1:
                thresholds = uniq
            else:
                thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for t in thresholds:
                left = vals <= t
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue

                left_mean = y[left].mean()
                right_mean = y[right].mean()

                pred = np.empty(n_samples, dtype=float)
                pred[left] = left_mean
                pred[right] = right_mean
                loss = np.mean((y - pred) ** 2)

                if loss < best_loss:
                    best_loss = loss
                    self.feature_index = j
                    self.threshold = float(t)
                    self.left_value = float(left_mean)
                    self.right_value = float(right_mean)

        return self

    def predict(self, X):
        vals = X[:, self.feature_index]
        out = np.empty(X.shape[0], dtype=float)
        left = vals <= self.threshold
        out[left] = self.left_value
        out[~left] = self.right_value
        return out


class GradientBoostingClassifierScratch:
    """Multiclass gradient boosting with regression stumps and softmax loss."""

    def __init__(self, n_estimators=40, learning_rate=0.1, random_state=None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state

        self.encoder_ = None
        self.classes_ = None
        self.stumps_ = None

    @staticmethod
    def _softmax(logits):
        z = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(z)
        return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1D and match X rows")

        self.encoder_ = _SimpleOneHotEncoder()
        Xn = self.encoder_.fit_transform(X)

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y], dtype=int)
        Y = np.zeros((len(y), K), dtype=float)
        Y[np.arange(len(y)), y_idx] = 1.0

        F = np.zeros((len(y), K), dtype=float)
        self.stumps_ = []

        for _ in range(self.n_estimators):
            P = self._softmax(F)
            residual = Y - P

            class_stumps = []
            for k in range(K):
                stump = _RegressionStump().fit(Xn, residual[:, k])
                update = stump.predict(Xn)
                F[:, k] += self.learning_rate * update
                class_stumps.append(stump)
            self.stumps_.append(class_stumps)

        return self

    def _check_is_fitted(self):
        if self.encoder_ is None or self.stumps_ is None:
            raise ValueError("Model is not fitted.")

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=object)
        Xn = self.encoder_.transform(X)

        K = len(self.classes_)
        F = np.zeros((Xn.shape[0], K), dtype=float)

        for class_stumps in self.stumps_:
            for k, stump in enumerate(class_stumps):
                F[:, k] += self.learning_rate * stump.predict(Xn)

        return self.classes_[np.argmax(F, axis=1)]

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)
