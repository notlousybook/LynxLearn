"""
ElasticNet Regression (L1 + L2 Regularization).
"""

import numpy as np
from ._base import BaseRegressor


class ElasticNet(BaseRegressor):
    """
    ElasticNet Regression with combined L1 and L2 regularization.

    Minimizes: 1/(2*n) * ||y - Xw||² + alpha * (l1_ratio * ||w||₁ + 0.5 * (1 - l1_ratio) * ||w||²)

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter. 0 <= l1_ratio <= 1.
        - l1_ratio=0: Ridge (L2 only)
        - l1_ratio=1: Lasso (L1 only)
        - 0 < l1_ratio < 1: ElasticNet (combination)
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.

    Attributes
    ----------
    weights : ndarray of shape (n_features,)
        Coefficients.
    bias : float
        Intercept.
    n_iter_ : int
        Actual iterations performed.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=True):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.n_iter_ = 0

    def _soft_threshold(self, x, gamma):
        """Soft thresholding operator."""
        if x > gamma:
            return x - gamma
        elif x < -gamma:
            return x + gamma
        else:
            return 0.0

    def train(self, X, y):
        """
        Train the ElasticNet regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ElasticNet
            The trained model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Center data if fitting intercept
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

        # Precompute X.T @ X diagonal
        XTX_diag = np.sum(X_centered ** 2, axis=0)

        # Coordinate descent
        alpha_l1 = self.alpha * self.l1_ratio * n_samples
        alpha_l2 = self.alpha * (1 - self.l1_ratio) * n_samples

        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()

            for j in range(n_features):
                # Compute partial residual
                residual = y_centered - X_centered @ self.weights + self.weights[j] * X_centered[:, j]

                # Update with both L1 and L2 regularization
                rho = np.dot(X_centered[:, j], residual)
                z = XTX_diag[j] + alpha_l2

                if z > 1e-10:
                    self.weights[j] = self._soft_threshold(rho, alpha_l1) / z

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
        return f"ElasticNet(alpha={self.alpha}, l1_ratio={self.l1_ratio}, max_iter={self.max_iter})"
