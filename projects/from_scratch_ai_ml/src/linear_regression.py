import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose

        self.weights = None  # shape: (n_features, 1)
        self.bias = None

        self.costs = []  # store loss history

    # -------------------------
    # Prediction: h(x) = XW + b
    # -------------------------
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    # -------------------------
    # Cost function (MSE)
    # J = 1/2m * sum((y_pred - y)^2)
    # -------------------------
    def _compute_cost(self, y, y_pred):
        m = y.shape[0]
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        return cost

    # -------------------------
    # Training (Gradient Descent)
    # -------------------------
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Ensure proper shapes
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Initialize parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        self.costs = []

        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self.predict(X)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute and store cost
            cost = self._compute_cost(y, y_pred)
            self.costs.append(cost)

            # Optional logging
            if self.verbose and i % 100 == 0:
                grad_norm = np.linalg.norm(dw)
                print(
                    f"Iter {i:04d} | Cost: {cost:.6f} | "
                    f"||dW||: {grad_norm:.6f} | bias: {self.bias:.4f}"
                )

        return self.costs
