import numpy as np


class _DecisionStump:
    """Weighted decision stump for numeric features."""

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y_signed, sample_weight):
        n_samples, n_features = X.shape
        best_err = float("inf")

        for j in range(n_features):
            vals = X[:, j]
            uniq = np.unique(vals)
            if len(uniq) == 1:
                thresholds = uniq
            else:
                thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for t in thresholds:
                pred = np.ones(n_samples)
                pred[vals <= t] = -1
                err = np.sum(sample_weight[pred != y_signed])

                polarity = 1
                if err > 0.5:
                    err = 1.0 - err
                    polarity = -1

                if err < best_err:
                    best_err = err
                    self.feature_index = j
                    self.threshold = float(t)
                    self.polarity = polarity

        return best_err

    def predict(self, X):
        n_samples = X.shape[0]
        pred = np.ones(n_samples)
        pred[X[:, self.feature_index] <= self.threshold] = -1
        return pred * self.polarity


class _SimpleOneHotEncoder:
    """Minimal mixed-type encoder to numeric matrix."""

    def __init__(self):
        self.is_numeric_ = None
        self.numeric_means_ = {}
        self.categories_ = {}
        self.n_features_in_ = None

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
        self.n_features_in_ = X.shape[1]
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
        parts = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if self.is_numeric_[j]:
                mean = self.numeric_means_[j]
                vals = np.array([float(v) if str(v).strip() != "" else np.nan for v in col], dtype=float)
                vals = np.where(np.isnan(vals), mean, vals)
                parts.append(vals.reshape(-1, 1))
            else:
                cats = self.categories_[j]
                cat_to_idx = {c: i for i, c in enumerate(cats)}
                out = np.zeros((X.shape[0], len(cats)), dtype=float)
                for i, v in enumerate(col):
                    key = str(v).strip() if str(v).strip() != "" else "MISSING"
                    if key in cat_to_idx:
                        out[i, cat_to_idx[key]] = 1.0
                parts.append(out)
        return np.hstack(parts) if parts else np.empty((X.shape[0], 0), dtype=float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class AdaBoostSAMMEScratch:
    """Multiclass AdaBoost (SAMME) with weighted decision stumps."""

    def __init__(self, n_estimators=30, random_state=None):
        self.n_estimators = int(n_estimators)
        self.random_state = random_state

        self.encoder_ = None
        self.classes_ = None
        self.class_to_idx_ = None
        self.learners_ = []
        self.alphas_ = []
        self.learner_classes_ = []

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
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        K = len(self.classes_)

        self.learners_ = []
        self.alphas_ = []
        self.learner_classes_ = []

        n = len(y)
        w = np.ones(n, dtype=float) / n
        eps = 1e-12

        for _ in range(self.n_estimators):
            best = None
            best_err = float("inf")
            best_class = None

            for c in self.classes_:
                y_bin = np.where(y == c, 1.0, -1.0)
                stump = _DecisionStump()
                err = stump.fit(Xn, y_bin, w)
                if err < best_err:
                    best_err = err
                    best = stump
                    best_class = c

            err = min(max(best_err, eps), 1.0 - eps)
            if err >= 1.0 - 1.0 / K:
                continue

            alpha = np.log((1.0 - err) / err) + np.log(K - 1.0)

            pred_bin = best.predict(Xn)
            pred_label = np.where(pred_bin > 0, best_class, None)
            incorrect = pred_label != y
            w *= np.exp(alpha * incorrect.astype(float))
            w /= np.sum(w)

            self.learners_.append(best)
            self.alphas_.append(float(alpha))
            self.learner_classes_.append(best_class)

        return self

    def _check_is_fitted(self):
        if self.encoder_ is None or len(self.learners_) == 0:
            raise ValueError("Model is not fitted.")

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=object)
        Xn = self.encoder_.transform(X)

        scores = np.zeros((Xn.shape[0], len(self.classes_)), dtype=float)
        for stump, alpha, c in zip(self.learners_, self.alphas_, self.learner_classes_):
            pred = stump.predict(Xn)
            class_idx = self.class_to_idx_[c]
            scores[:, class_idx] += alpha * (pred > 0).astype(float)

        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
