"""
Quantile Regression - Regression for specific quantiles.
"""

import numpy as np
from ._base import BaseRegressor


class QuantileRegressor(BaseRegressor):
    """
    Linear regression modeling a specific quantile (e.g., median).

    Unlike OLS which minimizes MSE (mean), quantile regression
    minimizes the pinball loss for a specific quantile.

    Parameters
    ----------
    quantile : float, default=0.5
        The quantile to model (0.5 = median, 0.9 = 90th percentile).
    alpha : float, default=0.0
        L1 regularization strength.
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    fit_intercept : bool, default=True
        Whether to fit intercept.

    Attributes
    ----------
    weights : ndarray
        Coefficients.
    bias : float
        Intercept.
    n_iter_ : int
        Actual iterations.
    """

    def __init__(self, quantile=0.5, alpha=0.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        super().__init__()
        self.quantile = quantile
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.n_iter_ = 0

    def _pinball_loss_gradient(self, residuals):
        """Compute gradient of pinball loss."""
        gradient = np.where(residuals >= 0, self.quantile, self.quantile - 1)
        return gradient

    def train(self, X, y):
        """
        Train the quantile regression using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : QuantileRegressor
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Initialize
        self.weights = np.zeros(n_features)
        self.bias = 0.0 if self.fit_intercept else 0.0

        # Simple gradient descent with subgradients
        learning_rate = 0.01
        best_loss = float('inf')
        patience = 50
        patience_counter = 0

        for iteration in range(self.max_iter):
            # Predictions
            y_pred = X @ self.weights + self.bias
            residuals = y - y_pred

            # Compute subgradients of pinball loss
            mask_positive = residuals >= 0
            n_positive = np.sum(mask_positive)
            n_negative = n_samples - n_positive

            # Handle edge cases where all residuals are positive or negative
            if n_positive == 0:
                # All residuals negative
                grad_weights = -np.mean(X * (self.quantile - 1), axis=0)
            elif n_negative == 0:
                # All residuals positive
                grad_weights = -np.mean(X * self.quantile, axis=0)
            else:
                grad_weights = -np.mean(X[mask_positive] * self.quantile, axis=0) \
                               - np.mean(X[~mask_positive] * (self.quantile - 1), axis=0)

            if self.fit_intercept:
                grad_bias = -np.mean(np.where(mask_positive, self.quantile, self.quantile - 1))

            # Add L1 regularization subgradient
            if self.alpha > 0:
                grad_weights += self.alpha * np.sign(self.weights)

            # Update
            self.weights -= learning_rate * grad_weights
            if self.fit_intercept:
                self.bias -= learning_rate * grad_bias

            # Compute loss for early stopping
            loss = np.mean(np.where(residuals >= 0,
                                   self.quantile * residuals,
                                   (self.quantile - 1) * residuals))

            if loss < best_loss - self.tol:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self._is_trained = True
        return self

    def __repr__(self):
        return f"QuantileRegressor(quantile={self.quantile}, alpha={self.alpha})"
