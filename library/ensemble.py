import numpy as np

from .nn import NeuralNetwork
from .random_forest import RandomForestClassifierScratch

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


class _NNClassifierMember:
    def __init__(self, hidden_layers, regularization=0.01, learning_rate=0.08, max_iterations=250):
        self.hidden_layers = list(hidden_layers)
        self.regularization = float(regularization)
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.model_ = None
        self.classes_ = None
        self.class_to_idx_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx_[v] for v in y], dtype=int)
        Y = np.zeros((len(y), len(self.classes_)), dtype=float)
        Y[np.arange(len(y)), y_idx] = 1.0

        layers = [X.shape[1]] + self.hidden_layers + [len(self.classes_)]
        self.model_ = NeuralNetwork(
            layers=layers,
            regularization=self.regularization,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
        ).fit(X, Y, normalize=True)
        return self

    def predict(self, X):
        probs = self.model_.predict(np.asarray(X, dtype=float), normalize=True)
        return self.classes_[np.argmax(probs, axis=1)]

class HeterogeneousBootstrapEnsembleEC3:
    def __init__(
        self,
        nn_architectures=None,
        n_rf_members=2,
        rf_params=None,
        nn_regularization=0.01,
        nn_learning_rate=0.08,
        nn_max_iterations=250,
        random_state=42,
    ):
        self.nn_architectures = nn_architectures or [[32], [24, 12], [48, 24]]
        self.n_rf_members = int(n_rf_members)
        self.rf_params = rf_params or {
            "n_trees": 15,
            "max_depth": 12,
            "min_size": 3,
            "min_gain": 1e-4,
        }
        self.nn_regularization = float(nn_regularization)
        self.nn_learning_rate = float(nn_learning_rate)
        self.nn_max_iterations = int(nn_max_iterations)
        self.random_state = random_state

        self.encoder_ = None
        self.members_ = []
        self.classes_ = None
        self._rng = np.random.default_rng(random_state)

    @staticmethod
    def _infer_numeric_cols(X):
        numeric_cols = []
        for j in range(X.shape[1]):
            ok = True
            for v in X[:, j]:
                s = str(v).strip()
                if s == "":
                    continue
                try:
                    float(s)
                except ValueError:
                    ok = False
                    break
            if ok:
                numeric_cols.append(j)
        return set(numeric_cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1D and match X rows")

        self.classes_ = np.unique(y)
        self.members_ = []

        # Shared encoder for the three NN members.
        self.encoder_ = _SimpleOneHotEncoder()
        X_encoded = self.encoder_.fit_transform(X)

        n = len(y)

        for arch in self.nn_architectures:
            boot_idx = self._rng.choice(n, size=n, replace=True)
            nn_member = _NNClassifierMember(
                hidden_layers=arch,
                regularization=self.nn_regularization,
                learning_rate=self.nn_learning_rate,
                max_iterations=self.nn_max_iterations,
            ).fit(X_encoded[boot_idx], y[boot_idx])
            self.members_.append(("nn", nn_member))

        for _ in range(self.n_rf_members):
            rf_boot_idx = self._rng.choice(n, size=n, replace=True)
            X_rf = X[rf_boot_idx]
            y_rf = y[rf_boot_idx]
            numeric_cols = self._infer_numeric_cols(X_rf)

            rf = RandomForestClassifierScratch(
                n_trees=self.rf_params.get("n_trees", 15),
                max_depth=self.rf_params.get("max_depth", 12),
                min_size=self.rf_params.get("min_size", 3),
                min_gain=self.rf_params.get("min_gain", 1e-4),
                numeric_cols=numeric_cols,
                random_state=self.random_state,
            ).fit(X_rf, y_rf)
            self.members_.append(("rf", rf))

        return self

    def _check_is_fitted(self):
        expected = len(self.nn_architectures) + self.n_rf_members
        if self.encoder_ is None or len(self.members_) != expected:
            raise ValueError("Model is not fitted.")

    def _majority_vote(self, row_preds):
        labels, counts = np.unique(row_preds, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=object)

        X_encoded = self.encoder_.transform(X)
        all_preds = []
        for member_type, member in self.members_:
            if member_type == "nn":
                all_preds.append(member.predict(X_encoded))
            else:
                all_preds.append(member.predict(X))

        stacked = np.vstack(all_preds).T
        return np.array([self._majority_vote(row) for row in stacked])

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)
