"""
Linear Support Vector Regression.
"""

import numpy as np

from ._base import BaseRegressor


class LinearSVR(BaseRegressor):
    """
    Linear Support Vector Regression.

    Uses the epsilon-insensitive loss function, which ignores errors
    smaller than epsilon. Good for robust regression.

    Parameters
    ----------
    epsilon : float, default=0.1
        Epsilon parameter in the epsilon-insensitive loss function.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C.
    loss : str, default='epsilon_insensitive'
        Loss function: 'epsilon_insensitive' or 'squared_epsilon_insensitive'.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for stopping criterion.
    learn_bias : bool, default=True
        Whether to learn the bias term.
        (Also accepts `fit_intercept` for backward compatibility)
    learning_rate : float, default=0.01
        Learning rate for gradient descent.

    Attributes
    ----------
    weights : ndarray
        Coefficients.
    bias : float
        Intercept.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        epsilon=0.1,
        C=1.0,
        loss="epsilon_insensitive",
        max_iter=1000,
        tol=1e-4,
        learn_bias=True,
        fit_intercept=None,
        learning_rate=0.01,
    ):
        super().__init__()
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        self.epsilon = epsilon
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.learn_bias = learn_bias
        self.fit_intercept = learn_bias  # Alias for backward compatibility
        self.learning_rate = learning_rate
        self.n_iter_ = 0

    def _epsilon_insensitive_loss_gradient(self, residual):
        """Compute gradient of epsilon-insensitive loss."""
        abs_residual = np.abs(residual)

        if self.loss == "epsilon_insensitive":
            # Subgradient: sign(residual) if |residual| > epsilon, else 0
            grad = np.where(abs_residual > self.epsilon, np.sign(residual), 0.0)
        elif self.loss == "squared_epsilon_insensitive":
            # Gradient: 2 * (residual - epsilon * sign(residual)) if |residual| > epsilon
            grad = np.where(
                abs_residual > self.epsilon,
                2 * (residual - self.epsilon * np.sign(residual)),
                0.0,
            )
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

        return grad

    def train(self, X, y):
        """
        Train the Linear SVR model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearSVR
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0 if self.fit_intercept else 0.0

        # Gradient descent
        prev_loss = float("inf")
        no_improvement_count = 0

        for iteration in range(self.max_iter):
            # Compute predictions
            y_pred = X @ self.weights + self.bias
            residuals = y_pred - y

            # Compute loss
            abs_residuals = np.abs(residuals)
            if self.loss == "epsilon_insensitive":
                loss = np.sum(np.maximum(0, abs_residuals - self.epsilon))
            else:  # squared_epsilon_insensitive
                loss = np.sum(np.maximum(0, abs_residuals - self.epsilon) ** 2)

            # Add L2 regularization
            loss += 0.5 * (1.0 / self.C) * np.sum(self.weights**2)

            # Compute gradients
            grad_loss = self._epsilon_insensitive_loss_gradient(residuals)

            # Gradient for weights
            grad_w = (
                np.mean(grad_loss[:, np.newaxis] * X, axis=0)
                + (1.0 / self.C) * self.weights
            )

            # Gradient for bias
            grad_b = np.mean(grad_loss)

            # Update parameters
            self.weights -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.bias -= self.learning_rate * grad_b

            # Check convergence
            if np.abs(prev_loss - loss) < self.tol:
                no_improvement_count += 1
                if no_improvement_count >= 5:
                    self.n_iter_ = iteration + 1
                    break
            else:
                no_improvement_count = 0

            prev_loss = loss

        self.n_iter_ = iteration + 1
        self._is_trained = True
        return self

    def __repr__(self):
        return f"LinearSVR(epsilon={self.epsilon}, C={self.C})"
