"""
Ridge Regression (L2 Regularization).
"""

import numpy as np
from ._base import BaseRegressor


class Ridge(BaseRegressor):
    """
    Ridge Regression with L2 regularization - prevents overfitting.

    Adds a penalty to large weights, making the model more stable
    and better at generalizing to new data.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Larger = simpler model (less overfitting).
    fit_intercept : bool, default=True
        Whether to learn the bias term.

    Examples
    --------
    >>> from lousybookml import Ridge
    >>> model = Ridge(alpha=0.5)  # Moderate regularization
    >>> model.train(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """

    def __init__(self, alpha=1.0, fit_intercept=True):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def train(self, X, y):
        """
        Train the Ridge regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : Ridge
            The trained model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        if self.fit_intercept:
            # Center X and y for intercept calculation
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean

            # Ridge solution for centered data
            I = np.eye(n_features)
            self.weights = (
                np.linalg.pinv(X_centered.T @ X_centered + self.alpha * I)
                @ X_centered.T
                @ y_centered
            )
            self.bias = y_mean - X_mean @ self.weights
        else:
            # No intercept: regularize all parameters
            X_b = X
            I = np.eye(n_features)
            theta = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
            self.weights = theta
            self.bias = 0.0

        self._is_trained = True
        return self

    def __repr__(self):
        return f"Ridge(alpha={self.alpha}, fit_intercept={self.fit_intercept})"