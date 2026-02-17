"""
Huber Regression - Robust regression with Huber loss.
"""

import numpy as np

from ._base import BaseRegressor


class HuberRegressor(BaseRegressor):
    """
    Linear regression with Huber loss for robustness to outliers.

    The Huber loss is less sensitive to outliers than MSE,
    switching from quadratic to linear loss at a threshold epsilon.

    Parameters
    ----------
    epsilon : float, default=1.35
        The parameter controlling the threshold for outlier detection.
        Smaller values make the regression more robust to outliers.
    alpha : float, default=0.0
        L2 regularization strength (Ridge-like).
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-5
        Convergence tolerance.
    learn_bias : bool, default=True
        Whether to learn the bias term.
        (Also accepts `fit_intercept` for backward compatibility)

    Attributes
    ----------
    weights : ndarray
        Coefficients.
    bias : float
        Intercept.
    n_iter_ : int
        Actual iterations.
    """

    def __init__(
        self,
        epsilon=1.35,
        alpha=0.0,
        max_iter=100,
        tol=1e-5,
        learn_bias=True,
        fit_intercept=None,
    ):
        super().__init__()
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learn_bias = learn_bias
        self.fit_intercept = learn_bias  # Alias for backward compatibility
        self.n_iter_ = 0

    def _huber_loss_gradient(self, residuals):
        """Compute Huber loss gradient."""
        abs_residuals = np.abs(residuals)
        mask = abs_residuals <= self.epsilon

        gradient = np.zeros_like(residuals)
        gradient[mask] = residuals[mask]
        gradient[~mask] = self.epsilon * np.sign(residuals[~mask])

        return gradient

    def _compute_weights(self, residuals):
        """Compute weights for weighted least squares."""
        abs_residuals = np.abs(residuals)
        weights = np.ones_like(residuals)

        mask = abs_residuals > self.epsilon
        weights[mask] = self.epsilon / abs_residuals[mask]

        return weights

    def train(self, X, y):
        """
        Train the Huber regression model using iteratively reweighted least squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : HuberRegressor
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Initialize with OLS
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y

        # Start with standard least squares
        self.weights = np.linalg.lstsq(X_centered, y_centered, rcond=None)[0]
        self.bias = 0.0

        # Iteratively reweighted least squares
        for iteration in range(self.max_iter):
            # Predictions
            y_pred = X_centered @ self.weights
            residuals = y_centered - y_pred

            # Compute weights based on residuals
            sample_weights = self._compute_weights(residuals)

            # Weighted least squares
            W = np.diag(sample_weights)
            XWX = X_centered.T @ W @ X_centered + self.alpha * np.eye(n_features)
            XWy = X_centered.T @ W @ y_centered

            weights_new = np.linalg.solve(XWX, XWy)

            # Check convergence
            if np.sum(np.abs(weights_new - self.weights)) < self.tol:
                self.weights = weights_new
                self.n_iter_ = iteration + 1
                break

            self.weights = weights_new
        else:
            self.n_iter_ = self.max_iter

        # Compute intercept
        if self.fit_intercept:
            self.bias = y_mean - X_mean @ self.weights

        self._is_trained = True
        return self

    def __repr__(self):
        return f"HuberRegressor(epsilon={self.epsilon}, alpha={self.alpha}, max_iter={self.max_iter})"
