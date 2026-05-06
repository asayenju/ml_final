import numpy as np


def sigmoid_activation(values):
    return 1.0 / (1.0 + np.exp(-values))


def sigmoid_derivative(linear_values):
    activation_values = sigmoid_activation(linear_values)
    return activation_values * (1.0 - activation_values)


def prepend_bias_column(values):
    if values.ndim == 1:
        return np.concatenate([np.ones(1), values])
    return np.concatenate([np.ones((values.shape[0], 1)), values], axis=1)


def run_forward_pass(features, weight_matrices):
    activations = [features]
    linear_outputs = []
    current_activation = features
    for weight_matrix in weight_matrices:
        activation_with_bias = prepend_bias_column(current_activation)
        linear_output = activation_with_bias @ weight_matrix.T
        current_activation = sigmoid_activation(linear_output)
        linear_outputs.append(linear_output)
        activations.append(current_activation)
    return activations, linear_outputs


def compute_cross_entropy_cost(true_labels, predicted_labels, weight_matrices, regularization_lambda):
    sample_count = true_labels.shape[0]
    eps = 1e-12
    clipped_predictions = np.clip(predicted_labels, eps, 1.0 - eps)
    cost = -np.sum(
        true_labels * np.log(clipped_predictions)
        + (1.0 - true_labels) * np.log(1.0 - clipped_predictions)
    ) / sample_count
    if regularization_lambda > 0:
        reg_sum = 0.0
        for weight_matrix in weight_matrices:
            reg_sum += np.sum(weight_matrix[:, 1:] ** 2)
        cost += (regularization_lambda / (2.0 * sample_count)) * reg_sum
    return cost


def run_backpropagation(features, labels, weight_matrices, reg_lambda):
    sample_count = features.shape[0]
    activations, linear_outputs = run_forward_pass(features, weight_matrices)
    deltas = [None] * len(weight_matrices)
    gradients = [None] * len(weight_matrices)

    deltas[-1] = activations[-1] - labels
    for layer_index in range(len(weight_matrices) - 2, -1, -1):
        next_weights = weight_matrices[layer_index + 1]
        deltas[layer_index] = (
            deltas[layer_index + 1] @ next_weights[:, 1:]
        ) * sigmoid_derivative(linear_outputs[layer_index])

    for layer_index, weight_matrix in enumerate(weight_matrices):
        previous_activation = prepend_bias_column(activations[layer_index])
        gradients[layer_index] = (deltas[layer_index].T @ previous_activation) / sample_count
        if reg_lambda > 0:
            regularization = (reg_lambda / sample_count) * weight_matrix
            regularization[:, 0] = 0.0
            gradients[layer_index] = gradients[layer_index] + regularization

    return activations, linear_outputs, deltas, gradients


def fit_min_max_normalization(features):
    min_values = np.min(features, axis=0)
    max_values = np.max(features, axis=0)
    ranges = max_values - min_values
    ranges[ranges == 0] = 1.0
    normalized = (features - min_values) / ranges
    return normalized, {"min": min_values, "range": ranges}


def apply_min_max_normalization(features, stats):
    return (features - stats["min"]) / stats["range"]


class NeuralNetwork:
    def __init__(
        self,
        layers,
        regularization=0.0,
        learning_rate=0.1,
    ):
        self.layer_sizes = list(layers)
        self.reg_lambda = regularization
        self.learning_rate = learning_rate
        self.weight_matrices = self._init_weight_matrices()
        self.normalization_stats = None

    def _init_weight_matrices(self):
        rng = np.random.default_rng()
        weight_matrices = []
        for layer_index in range(1, len(self.layer_sizes)):
            rows = self.layer_sizes[layer_index]
            cols = self.layer_sizes[layer_index - 1] + 1
            weight_matrix = rng.uniform(-1.0, 1.0, size=(rows, cols))
            weight_matrices.append(weight_matrix)
        return weight_matrices

    def fit(self, features, labels, normalize=True):
        if normalize:
            features, self.normalization_stats = fit_min_max_normalization(features)
        previous_cost = None
        stopping_epsilon =0.0001
        while True:
            activations, _, _, gradients = run_backpropagation(
                features,
                labels,
                self.weight_matrices,
                self.reg_lambda,
            )
            for layer_index in range(len(self.weight_matrices)):
                self.weight_matrices[layer_index] = (
                    self.weight_matrices[layer_index]
                    - self.learning_rate * gradients[layer_index]
                )
            cost = compute_cross_entropy_cost(
                labels,
                activations[-1],
                self.weight_matrices,
                self.reg_lambda,
            )
            if previous_cost is not None and abs(previous_cost - cost) < stopping_epsilon:
                break
            previous_cost = cost
        return self

    def predict(self, features, normalize=True):
        if normalize and self.normalization_stats is not None:
            features = apply_min_max_normalization(features, self.normalization_stats)
        activations, _ = run_forward_pass(features, self.weight_matrices)
        return activations[-1]
