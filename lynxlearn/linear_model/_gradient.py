"""
Linear Regression using Gradient Descent.
"""

import numpy as np

from ._base import BaseRegressor


class GradientDescentRegressor(BaseRegressor):
    """
    Linear Regression using Gradient Descent - learns by iteration.

    Good for large datasets where Normal Equation is too slow.
    Shows progress through cost history.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for each iteration. Too high = unstable, too low = slow.
    n_iterations : int, default=1000
        Maximum training steps.
    tolerance : float, default=1e-6
        Stop early if improvement is smaller than this.
    learn_bias : bool, default=True
        Whether to learn the bias term.
        (Also accepts `fit_intercept` for backward compatibility)

    Attributes
    ----------
    weights : ndarray
        Learned coefficients.
    bias : float
        Learned intercept.
    cost_history : list
        Training error at each step (useful for plotting).
    n_iter_ : int
        Actual iterations run.

    Examples
    --------
    >>> model = GradientDescentRegressor(learning_rate=0.01)
    >>> model.train(X_train, y_train)
    >>> print(f"Trained for {model.n_iter_} iterations")
    """

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=1000,
        tolerance=1e-6,
        learn_bias=True,
        fit_intercept=None,
    ):
        super().__init__()
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.learn_bias = learn_bias
        self.fit_intercept = learn_bias  # Alias for backward compatibility
        self.cost_history = []
        self.n_iter_ = 0

    def _compute_cost(self, X, y):
        """Compute Mean Squared Error cost."""
        n_samples = X.shape[0]
        predictions = X @ self.weights + self.bias
        cost = (1 / (2 * n_samples)) * np.sum((predictions - y) ** 2)
        return cost

    def train(self, X, y):
        """
        Train the model using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GradientDescentRegressor
            The trained model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0 if self.fit_intercept else 0.0
        self.cost_history = []

        for i in range(self.n_iterations):
            # Forward pass: compute predictions
            predictions = X @ self.weights + self.bias

            # Compute gradients
            error = predictions - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias -= self.learning_rate * db

            # Compute and store cost
            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)

            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - cost) < self.tolerance:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.n_iterations

        self._is_trained = True
        return self

    def __repr__(self):
        return (
            f"GradientDescentRegressor("
            f"learning_rate={self.learning_rate}, "
            f"n_iterations={self.n_iterations}, "
            f"tolerance={self.tolerance}, "
            f"fit_intercept={self.fit_intercept})"
        )
