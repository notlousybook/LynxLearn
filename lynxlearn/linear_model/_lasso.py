"""
Lasso Regression (L1 Regularization) using coordinate descent.
"""

import numpy as np
from ._base import BaseRegressor


class Lasso(BaseRegressor):
    """
    Lasso Regression with L1 regularization - automatically selects important features.

    Sets unimportant feature weights to exactly zero, effectively doing
    automatic feature selection. Great when you have many features but
    only a few are actually important.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Larger = more features set to zero.
    max_iter : int, default=1000
        Maximum training iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    learn_bias : bool, default=True
        Whether to learn the bias term.
        (Also accepts `fit_intercept` for backward compatibility)

    Examples
    --------
    >>> from lousybookml import Lasso
    >>> model = Lasso(alpha=0.1)
    >>> model.train(X_train, y_train)
    >>> print(f"Non-zero features: {np.sum(model.weights != 0)}")
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, learn_bias=True, fit_intercept=None):
        super().__init__()
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learn_bias = learn_bias
        self.fit_intercept = learn_bias  # Alias for backward compatibility
        self.n_iter_ = 0

    def _soft_threshold(self, x, gamma):
        """Soft thresholding operator for L1 regularization."""
        if x > gamma:
            return x - gamma
        elif x < -gamma:
            return x + gamma
        else:
            return 0.0

    def train(self, X, y):
        """
        Train the Lasso regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : Lasso
            The trained model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Center X and y if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y

        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0 if self.fit_intercept else 0.0

        # Precompute X.T @ X diagonal for speed
        XTX_diag = np.sum(X_centered ** 2, axis=0)

        # Coordinate descent
        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()

            for j in range(n_features):
                # Compute partial residual
                residual = y_centered - X_centered @ self.weights + self.weights[j] * X_centered[:, j]

                # Update weight j
                rho = np.dot(X_centered[:, j], residual)
                z = XTX_diag[j]

                if z > 1e-10:  # Avoid division by zero
                    self.weights[j] = self._soft_threshold(rho, self.alpha * n_samples) / z

            # Check convergence
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        # Compute intercept
        if self.fit_intercept:
            self.bias = y_mean - X_mean @ self.weights

        self._is_trained = True
        return self

    def __repr__(self):
        return f"Lasso(alpha={self.alpha}, max_iter={self.max_iter}, tol={self.tol})"
